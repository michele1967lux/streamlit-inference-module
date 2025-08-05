"""
User Manager for Streamlit Inference Module
===========================================

Handles user authentication, session management, and multi-user support.
"""

import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import secrets

logger = logging.getLogger(__name__)


class UserManager:
    """Manages user authentication and sessions."""
    
    def __init__(self, db_path: Path):
        """Initialize user manager."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Session timeout (hours)
        self.session_timeout = 24
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for users."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    role TEXT DEFAULT 'user'
                )
                ''')
                
                # Sessions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
                ''')
                
                # User preferences table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(username, preference_key),
                    FOREIGN KEY (username) REFERENCES users (username)
                )
                ''')
                
                conn.commit()
                logger.info("User database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing user database: {e}")
            raise
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def register_user(self, username: str, password: str, role: str = 'user') -> bool:
        """Register a new user."""
        try:
            # Validate input
            if not username or not password:
                logger.warning("Username and password are required")
                return False
            
            if len(username) < 3 or len(password) < 6:
                logger.warning("Username must be at least 3 characters, password at least 6")
                return False
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user already exists
                cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    logger.warning(f"User {username} already exists")
                    return False
                
                # Insert new user
                cursor.execute('''
                INSERT INTO users (username, password_hash, salt, created_at, role)
                VALUES (?, ?, ?, ?, ?)
                ''', (username, password_hash, salt, datetime.now().isoformat(), role))
                
                conn.commit()
                logger.info(f"User {username} registered successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get user data
                cursor.execute('''
                SELECT password_hash, salt, is_active 
                FROM users 
                WHERE username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"User {username} not found")
                    return False
                
                stored_hash, salt, is_active = result
                
                if not is_active:
                    logger.warning(f"User {username} is inactive")
                    return False
                
                # Verify password
                password_hash, _ = self._hash_password(password, salt)
                
                if password_hash == stored_hash:
                    # Update last login
                    cursor.execute('''
                    UPDATE users 
                    SET last_login = ? 
                    WHERE username = ?
                    ''', (datetime.now().isoformat(), username))
                    
                    # Create session
                    self._create_session(username, conn)
                    
                    conn.commit()
                    logger.info(f"User {username} authenticated successfully")
                    return True
                else:
                    logger.warning(f"Invalid password for user {username}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return False
    
    def _create_session(self, username: str, conn: sqlite3.Connection) -> str:
        """Create a new session for user."""
        session_token = secrets.token_urlsafe(32)
        now = datetime.now().isoformat()
        
        cursor = conn.cursor()
        
        # Deactivate old sessions
        cursor.execute('''
        UPDATE sessions 
        SET is_active = 0 
        WHERE username = ? AND is_active = 1
        ''', (username,))
        
        # Create new session
        cursor.execute('''
        INSERT INTO sessions (username, session_token, created_at, last_activity)
        VALUES (?, ?, ?, ?)
        ''', (username, session_token, now, now))
        
        return session_token
    
    def get_active_users(self) -> List[str]:
        """Get list of currently active users."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up expired sessions first
                self._cleanup_expired_sessions(conn)
                
                # Get active users
                cursor.execute('''
                SELECT DISTINCT username 
                FROM sessions 
                WHERE is_active = 1 
                AND datetime(last_activity) > datetime('now', '-1 day')
                ''')
                
                users = [row[0] for row in cursor.fetchall()]
                return users
                
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    def _cleanup_expired_sessions(self, conn: sqlite3.Connection):
        """Clean up expired sessions."""
        cursor = conn.cursor()
        
        # Deactivate sessions older than timeout
        cutoff_time = datetime.now() - timedelta(hours=self.session_timeout)
        
        cursor.execute('''
        UPDATE sessions 
        SET is_active = 0 
        WHERE is_active = 1 
        AND datetime(last_activity) < ?
        ''', (cutoff_time.isoformat(),))
    
    def logout_user(self, username: str) -> bool:
        """Logout user and deactivate sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE sessions 
                SET is_active = 0 
                WHERE username = ? AND is_active = 1
                ''', (username,))
                
                conn.commit()
                logger.info(f"User {username} logged out")
                return True
                
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False
    
    def update_session_activity(self, username: str) -> bool:
        """Update last activity for user session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE sessions 
                SET last_activity = ? 
                WHERE username = ? AND is_active = 1
                ''', (datetime.now().isoformat(), username))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
            return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, any]]:
        """Get user information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT username, created_at, last_login, role
                FROM users 
                WHERE username = ? AND is_active = 1
                ''', (username,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        "username": result[0],
                        "created_at": result[1],
                        "last_login": result[2],
                        "role": result[3]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    def set_user_preference(self, username: str, key: str, value: str) -> bool:
        """Set user preference."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (username, preference_key, preference_value, updated_at)
                VALUES (?, ?, ?, ?)
                ''', (username, key, value, datetime.now().isoformat()))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error setting user preference: {e}")
            return False
    
    def get_user_preference(self, username: str, key: str, default: str = None) -> Optional[str]:
        """Get user preference."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT preference_value 
                FROM user_preferences 
                WHERE username = ? AND preference_key = ?
                ''', (username, key))
                
                result = cursor.fetchone()
                return result[0] if result else default
                
        except Exception as e:
            logger.error(f"Error getting user preference: {e}")
            return default
    
    def get_all_user_preferences(self, username: str) -> Dict[str, str]:
        """Get all user preferences."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT preference_key, preference_value 
                FROM user_preferences 
                WHERE username = ?
                ''', (username,))
                
                return dict(cursor.fetchall())
                
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    def delete_user(self, username: str) -> bool:
        """Delete user account (admin only)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Deactivate instead of deleting to preserve data integrity
                cursor.execute('''
                UPDATE users 
                SET is_active = 0 
                WHERE username = ?
                ''', (username,))
                
                # Deactivate sessions
                cursor.execute('''
                UPDATE sessions 
                SET is_active = 0 
                WHERE username = ?
                ''', (username,))
                
                conn.commit()
                logger.info(f"User {username} deactivated")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False
    
    def get_user_stats(self) -> Dict[str, any]:
        """Get user statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total users
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                total_users = cursor.fetchone()[0]
                
                # Active users
                active_users = len(self.get_active_users())
                
                # Recent registrations (last 7 days)
                cursor.execute('''
                SELECT COUNT(*) FROM users 
                WHERE is_active = 1 
                AND datetime(created_at) > datetime('now', '-7 days')
                ''')
                recent_registrations = cursor.fetchone()[0]
                
                return {
                    "total_users": total_users,
                    "active_users": active_users,
                    "recent_registrations": recent_registrations,
                    "max_users": 5
                }
                
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {
                "total_users": 0,
                "active_users": 0,
                "recent_registrations": 0,
                "max_users": 5
            }