
from typing import Tuple, List, Optional
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import face_recognition
from datetime import datetime, date, time
import sqlite3
sqlite3.register_adapter(datetime, lambda val: val.isoformat())
sqlite3.register_adapter(date, lambda val: val.isoformat())
from PIL import Image
import plotly.express as px
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttendanceSystem:
    def __init__(self):
        """Initialize the attendance system with database connection and required directories"""
        try:
            self.conn = sqlite3.connect('attendance.db', check_same_thread=False)
            self.setup_directories()
            self.create_tables()
            logger.info("Attendance system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing attendance system: {e}")
            raise

    def setup_directories(self):
        """Create necessary directories for storing images"""
        try:
            directories = ['images/profiles', 'images/logs']
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info("Directories created successfully")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise

    def create_tables(self):
        """Create necessary database tables"""
        try:
            cursor = self.conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL,
                    employee_id TEXT UNIQUE NOT NULL,
                    face_encoding BLOB NOT NULL,
                    profile_image TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    check_in TIMESTAMP,
                    check_out TIMESTAMP,
                    date DATE NOT NULL,
                    status TEXT NOT NULL,
                    confidence_score REAL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def register_user(self, full_name: str, email: str, department: str, 
                     employee_id: str, image_file) -> Tuple[bool, str]:
        """Register a new user with face encoding"""
        try:
            # Input validation
            if not all([full_name, email, department, employee_id, image_file]):
                return False, "All fields are required"

            # Check if user already exists
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM users WHERE email = ? OR employee_id = ?', 
                         (email, employee_id))
            if cursor.fetchone():
                return False, "User with this email or employee ID already exists"

            # Process and save profile image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"images/profiles/{employee_id}_{timestamp}.jpg"
            
            # Convert and save image
            image = Image.open(image_file)
            image.save(image_path)
            
            # Generate face encoding
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)
            
            if not face_encodings:
                os.remove(image_path)
                return False, "No face detected in the image"
            
            face_encoding = face_encodings[0]
            
            # Save user data
            cursor.execute('''
                INSERT INTO users (full_name, email, department, employee_id, 
                                 face_encoding, profile_image)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (full_name, email, department, employee_id, 
                  face_encoding.tobytes(), image_path))
            
            self.conn.commit()
            logger.info(f"User {full_name} registered successfully")
            return True, "User registered successfully"
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            if 'image_path' in locals():
                os.remove(image_path)
            return False, f"Registration failed: {str(e)}"

    def get_user_data(self) -> List[Tuple]:
        """Retrieve all active users with their face encodings"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, full_name, face_encoding 
                FROM users 
                WHERE status = 'active'
            ''')
            users = cursor.fetchall()
            
            # Convert face encodings back to numpy arrays
            return [(id, name, np.frombuffer(encoding, dtype=np.float64)) 
                    for id, name, encoding in users]
        except Exception as e:
            logger.error(f"Error retrieving user data: {e}")
            return []
    
    # Add these methods to your AttendanceSystem class

    def get_recent_attendance(self) -> pd.DataFrame:
        """Get recent attendance records"""
        try:
            query = '''
                SELECT 
                    u.full_name,
                    a.date,
                    a.check_in,
                    a.check_out,
                    a.status,
                    a.confidence_score
                FROM attendance a
                JOIN users u ON a.user_id = u.id
                WHERE a.date >= date('now', '-7 days')
                ORDER BY a.date DESC, a.check_in DESC
                LIMIT 10
            '''
            df = pd.read_sql_query(query, self.conn)
            
            # Format datetime columns
            if not df.empty:
                df['check_in'] = pd.to_datetime(df['check_in']).dt.strftime('%I:%M %p')
                df['check_out'] = pd.to_datetime(df['check_out']).dt.strftime('%I:%M %p')
                df['confidence_score'] = df['confidence_score'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
            
            return df
        except Exception as e:
            logger.error(f"Error getting recent attendance: {e}")
            return pd.DataFrame()

    def get_attendance_report(self, start_date, end_date) -> pd.DataFrame:
        """Get attendance report for a date range"""
        try:
            query = '''
                SELECT 
                    a.date,
                    COUNT(DISTINCT a.user_id) as attendance_count,
                    GROUP_CONCAT(DISTINCT u.full_name) as present_users,
                    AVG(a.confidence_score) as avg_confidence
                FROM attendance a
                JOIN users u ON a.user_id = u.id
                WHERE a.date BETWEEN ? AND ?
                GROUP BY a.date
                ORDER BY a.date
            '''
            
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            
            if not df.empty:
                # Calculate attendance rate
                total_users = self.get_attendance_stats()['total_users']
                df['attendance_rate'] = (df['attendance_count'] / total_users * 100).round(2)
                
                # Format confidence score
                df['avg_confidence'] = df['avg_confidence'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception as e:
            logger.error(f"Error getting attendance report: {e}")
            return pd.DataFrame()

    def get_department_attendance(self, start_date, end_date) -> pd.DataFrame:
        """Get department-wise attendance statistics"""
        try:
            query = '''
                SELECT 
                    u.department,
                    COUNT(DISTINCT a.user_id) as present_count,
                    COUNT(DISTINCT u.id) as total_users,
                    CAST(COUNT(DISTINCT a.user_id) * 100.0 / COUNT(DISTINCT u.id) AS REAL) as attendance_rate
                FROM users u
                LEFT JOIN attendance a ON u.id = a.user_id 
                    AND a.date BETWEEN ? AND ?
                WHERE u.status = 'active'
                GROUP BY u.department
            '''
            
            return pd.read_sql_query(query, self.conn, params=(start_date, end_date))
        except Exception as e:
            logger.error(f"Error getting department attendance: {e}")
            return pd.DataFrame()
        
        # Add these new methods to the AttendanceSystem class

    def reset_user_data(self, user_id: int) -> Tuple[bool, str]:
        """Reset attendance data for a specific user"""
        try:
            cursor = self.conn.cursor()
            
            # Get user info before deletion
            cursor.execute('SELECT full_name FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                return False, "User not found"
                
            # Delete attendance records
            cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
            
            self.conn.commit()
            logger.info(f"Reset attendance data for user {user[0]}")
            return True, f"Successfully reset attendance data for {user[0]}"
            
        except Exception as e:
            logger.error(f"Error resetting user data: {e}")
            return False, f"Failed to reset user data: {str(e)}"

    def reset_all_attendance(self) -> Tuple[bool, str]:
        """Reset all attendance records"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM attendance')
            self.conn.commit()
            logger.info("Reset all attendance records")
            return True, "Successfully reset all attendance records"
            
        except Exception as e:
            logger.error(f"Error resetting all attendance: {e}")
            return False, f"Failed to reset attendance data: {str(e)}"

    def get_all_users(self) -> pd.DataFrame:
        """Get list of all active users"""
        try:
            query = '''
                SELECT 
                    id,
                    full_name,
                    email,
                    department,
                    employee_id,
                    created_at
                FROM users
                WHERE status = 'active'
                ORDER BY full_name
            '''
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Error getting user list: {e}")
            return pd.DataFrame()

    def mark_attendance(self, user_id: int, name: str, confidence: float):
        """Record attendance entry"""
        try:
            cursor = self.conn.cursor()
            current_date = date.today()
            current_time = datetime.now()
            
            # Check for existing attendance record
            cursor.execute('''
                SELECT id, check_in 
                FROM attendance 
                WHERE user_id = ? AND date = ? AND check_out IS NULL
            ''', (user_id, current_date))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update check-out time for existing record
                entry_id = existing[0]
                cursor.execute('''
                    UPDATE attendance 
                    SET check_out = ? 
                    WHERE id = ?
                ''', (current_time, entry_id))
            else:
                # Create new attendance record
                cursor.execute('''
                    INSERT INTO attendance 
                    (user_id, check_in, date, status, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, current_time, current_date, 'present', confidence))
            
            self.conn.commit()
            logger.info(f"Attendance marked for user {name}")
            
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            raise

    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """Delete a user and all their associated data"""
        try:
            cursor = self.conn.cursor()
            
            # Get user info and image path before deletion
            cursor.execute('SELECT full_name, profile_image FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                return False, "User not found"
                
            # Delete user's profile image
            if os.path.exists(user[1]):
                os.remove(user[1])
                
            # Delete attendance records
            cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
            
            # Delete user
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            
            self.conn.commit()
            logger.info(f"Deleted user {user[0]} and associated data")
            return True, f"Successfully deleted user {user[0]} and all associated data"
            
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False, f"Failed to delete user: {str(e)}"

    def delete_all_users(self) -> Tuple[bool, str]:
        """Delete all users and associated data"""
        try:
            cursor = self.conn.cursor()
            
            # Get all profile image paths
            cursor.execute('SELECT profile_image FROM users')
            image_paths = cursor.fetchall()
            
            # Delete all profile images
            for (image_path,) in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            # Delete all attendance records
            cursor.execute('DELETE FROM attendance')
            
            # Delete all users
            cursor.execute('DELETE FROM users')
            
            self.conn.commit()
            logger.info("Deleted all users and associated data")
            return True, "Successfully deleted all users and associated data"
            
        except Exception as e:
            logger.error(f"Error deleting all users: {e}")
            return False, f"Failed to delete all users: {str(e)}"

    def get_attendance_stats(self) -> dict:
        """Get summary statistics for attendance"""
        try:
            cursor = self.conn.cursor()
            today = date.today()
            
            # Get total users
            cursor.execute('SELECT COUNT(*) FROM users WHERE status = "active"')
            total_users = cursor.fetchone()[0]
            
            # Get today's attendance
            cursor.execute('''
                SELECT COUNT(DISTINCT user_id) 
                FROM attendance 
                WHERE date = ?
            ''', (today,))
            today_attendance = cursor.fetchone()[0]
            
            # Get weekly attendance average
            cursor.execute('''
                SELECT AVG(daily_count) 
                FROM (
                    SELECT date, COUNT(DISTINCT user_id) as daily_count
                    FROM attendance
                    WHERE date >= date('now', '-7 days')
                    GROUP BY date
                )
            ''')
            weekly_avg = cursor.fetchone()[0] or 0
            
            return {
                'total_users': total_users,
                'today_attendance': today_attendance,
                'weekly_average': round(weekly_avg, 2),
                'attendance_rate': round((today_attendance/total_users * 100) if total_users > 0 else 0, 2)
            }
        except Exception as e:
            logger.error(f"Error getting attendance stats: {e}")
            return {'total_users': 0, 'today_attendance': 0, 'weekly_average': 0, 'attendance_rate': 0}

    def process_face_recognition(self, frame, known_face_encodings, known_face_data):
        """Process video frame for face recognition"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, 
                    face_encoding,
                    tolerance=0.6
                )
                name = "Unknown"
                confidence = 0.0
                user_id = None

                if True in matches:
                    # Find best match
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]
                    
                    if confidence > 0.5:
                        user_id, name, _ = known_face_data[best_match_index]
                        self.mark_attendance(user_id, name, confidence)
                        
                face_names.append((name, confidence))
            
            return face_locations, face_names
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return [], []

   # Update the streamlit_interface method to include the About page

    def streamlit_interface(self):
        """Main Streamlit interface"""
        st.set_page_config(
            page_title="Face Recognition Attendance",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom styling
        st.markdown("""
            <style>
            .main {
                padding: 1rem;
            }
            .stButton>button {
                width: 100%;
                background-color: #4CAF50;
                color: white;
            }
            .developer-card {
                padding: 1.5rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
                margin-bottom: 1rem;
            }
            .reset-button {
                background-color: #dc3545 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            page = st.radio(
                "Select Page",
                ["Dashboard", "Register User", "Take Attendance", "Reports", "Reset Data", "About"]
            )

        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Register User":
            self.show_registration()
        elif page == "Take Attendance":
            self.show_attendance()
        elif page == "Reports":
            self.show_reports()
        elif page == "Reset Data":
            self.show_reset_data()
        elif page == "About":
            self.show_about()

    def show_reset_data(self):
        
        """Reset Data page with reset and delete options"""
        st.title("Reset Data")
        
        # Custom CSS for tabs
        st.markdown("""
            <style>
            /* Custom styling for tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 20px;
                padding: 0.5rem;
                background-color: #f8f9fa;
                border-radius: 10px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding: 0 20px;
                color: #6c757d;
                border-radius: 5px;
                background-color: white;
                border: 2px solid #dee2e6;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #228BE6;
                color: white !important;
                border-color: #228BE6;
                font-weight: bold;
            }
            
            /* Hover effect for tabs */
            .stTabs [data-baseweb="tab"]:hover {
                color: #228BE6;
                border-color: #228BE6;
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create tabs for different reset options
        reset_tab, delete_tab = st.tabs(["🔄 Reset Attendance", "🗑️ Delete Users"])
        
        with reset_tab:
            st.header("Reset Attendance Records")
            
            # Reset Individual User Attendance
            st.subheader("Reset Individual User Attendance")
            users_df = self.get_all_users()
            
            if not users_df.empty:
                selected_user = st.selectbox(
                    "Select User",
                    options=users_df['id'].tolist(),
                    format_func=lambda x: users_df[users_df['id'] == x]['full_name'].iloc[0],
                    key="reset_attendance_user"
                )
                
                if st.button("Reset Selected User Attendance", key="reset_user"):
                    if st.session_state.get('confirm_reset_user', False):
                        success, message = self.reset_user_data(selected_user)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                        st.session_state['confirm_reset_user'] = False
                    else:
                        st.warning("Are you sure you want to reset this user's attendance data? Click again to confirm.")
                        st.session_state['confirm_reset_user'] = True
            
            # Reset All Attendance
            st.subheader("Reset All Attendance Records")
            if st.button("Reset All Attendance", key="reset_all"):
                if st.session_state.get('confirm_reset_all', False):
                    success, message = self.reset_all_attendance()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.session_state['confirm_reset_all'] = False
                else:
                    st.warning("⚠️ WARNING: This will delete ALL attendance records. This action cannot be undone. Click again to confirm.")
                    st.session_state['confirm_reset_all'] = True
        
        with delete_tab:
            st.header("Delete Users")
            
            # Delete Individual User
            st.subheader("Delete Individual User")
            users_df = self.get_all_users()
            
            if not users_df.empty:
                selected_user = st.selectbox(
                    "Select User",
                    options=users_df['id'].tolist(),
                    format_func=lambda x: users_df[users_df['id'] == x]['full_name'].iloc[0],
                    key="delete_user"
                )
                
                if st.button("Delete Selected User", key="delete_user_btn"):
                    if st.session_state.get('confirm_delete_user', False):
                        success, message = self.delete_user(selected_user)
                        if success:
                            st.success(message)
                            st.rerun()  # Refresh the page to update user list
                        else:
                            st.error(message)
                        st.session_state['confirm_delete_user'] = False
                    else:
                        st.warning("⚠️ WARNING: This will permanently delete the user and all their data. Click again to confirm.")
                        st.session_state['confirm_delete_user'] = True
            
            # Delete All Users
            st.subheader("Delete All Users")
            if st.button("Delete All Users", key="delete_all"):
                if st.session_state.get('confirm_delete_all', False):
                    success, message = self.delete_all_users()
                    if success:
                        st.success(message)
                        st.rerun()  # Refresh the page
                    else:
                        st.error(message)
                    st.session_state['confirm_delete_all'] = False
                else:
                    st.warning("⚠️ WARNING: This will permanently delete ALL users and their data. This action cannot be undone. Click again to confirm.")
                    st.session_state['confirm_delete_all'] = True
    def show_about(self):
        """About page with developer information and usage instructions"""
        st.title("About")
        
        # Custom CSS for about tabs and cards (CSS remains the same as before)
        st.markdown("""
            <style>
            /* Custom styling for about tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 20px;
                padding: 0.5rem;
                background-color: #f0f8ff;
                border-radius: 10px;
            }
            
            /* Style for both tabs */
            .stTabs [data-baseweb="tab"] {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0 30px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            /* Hover effects */
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #357abd;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Selected tab styles */
            .stTabs [aria-selected="true"] {
                background-color: #357abd;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Unified card style for all information boxes */
            .info-card {
                background: linear-gradient(145deg, #ffffff, #f0f0f0);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 1.5rem;
                box-shadow: 5px 5px 15px #d1d1d1, -5px -5px 15px #ffffff;
                transition: transform 0.3s ease;
            }
            
            .info-card:hover {
                transform: translateY(-5px);
            }
            
            .info-card h3 {
                color: #4a90e2;
                margin-bottom: 1rem;
                font-size: 1.5rem;
            }
            
            .info-card p {
                color: #2c3e50;
                margin-bottom: 0.5rem;
                font-size: 1rem;
            }
            
            .info-card strong {
                color: #357abd;
            }
            
            /* Section headers */
            .section-header {
                color: #2c3e50;
                border-left: 5px solid #4a90e2;
                padding-left: 1rem;
                margin: 2rem 0 1rem 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Add tabs with icons
        about_tab, usage_tab = st.tabs(["👨‍💻 About Developers", "📚 How to Use"])

        with about_tab:
            st.header("About the Developer")
            col1, col2 = st.columns(2)

            # First row of developers
            with col1:
                st.markdown(
                    """
                    <div class="info-card" style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; background: #f9f9f9; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h3 style="color: #007BFF; font-family: 'Poppins', sans-serif;">🚀 Meet the Developer</h3>
                        <p><strong style="font-size: 18px;">Talha Iqbal</strong></p>
                        <p style="margin: 10px 0; line-height: 1.5;">
                            Talha is a passionate developer with a focus on crafting innovative web and AI solutions.
                            As a student at <strong>Air University, Islamabad</strong>, he combines his academic
                            knowledge with hands-on experience to build smart, impactful projects.
                        </p>
                    </div>

                    """,
                    unsafe_allow_html=True,
                )

                
            
            # System Information
            st.header("System Information")
            st.markdown("""
            <div class="info-card">
                <h3>Technical Details</h3>
                <p><strong>Framework:</strong> Python + Streamlit</p>
                <p><strong>Features:</strong></p>
                <p>• Real-time Face Recognition</p>
                <p>• Attendance Tracking</p>
                <p>• Reporting & Analytics</p>
                <p>• User Management</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage tab content remains the same
        with usage_tab:
            st.header("How to Use the Application")
            
            sections = [
                ("1. User Registration", """
                • Navigate to the 'Register User' page
                • Fill in all required fields (Full Name, Email, Department, Employee ID)
                • Upload a clear face photo of the user
                • Click 'Register' to add the user to the system
                • Ensure the photo has good lighting and shows the face clearly
                """),
                ("2. Taking Attendance", """
                • Go to the 'Take Attendance' page
                • Click 'Start Camera' to begin face recognition
                • Users should face the camera directly
                • The system will automatically recognize registered faces
                • Green boxes indicate successful recognition
                • Red boxes indicate unrecognized faces
                • Click 'Stop' when finished
                """),
                ("3. Viewing Reports", """
                • Access the 'Reports' page
                • Select date range for attendance data
                • View attendance trends and detailed reports
                • Download reports in CSV format for further analysis
                • Monitor attendance rates and patterns
                """),
                ("4. Managing Data", """
                • Use the 'Reset Data' page for data management
                • Reset attendance records for individual users
                • Reset all attendance records while keeping user profiles
                • Delete individual users and their data
                • Delete all users and system data if needed
                • Always confirm before performing delete operations
                """),
                ("5. Dashboard", """
                • Monitor key metrics on the dashboard
                • View total users and today's attendance
                • Check attendance rates and weekly averages
                • See recent attendance records
                • Track department-wise attendance
                """)
            ]
            
            for title, content in sections:
                st.markdown(f"""
                <div class="info-card">
                    <h3>{title}</h3>
                    {content}
                </div>
                """, unsafe_allow_html=True)
                

    def show_dashboard(self):
        """Dashboard page"""
        st.title("Attendance Dashboard")
        
        # Get statistics
        stats = self.get_attendance_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", stats['total_users'])
        with col2:
            st.metric("Today's Attendance", stats['today_attendance'])
        with col3:
            st.metric("Attendance Rate", f"{stats['attendance_rate']}%")
        with col4:
            st.metric("Weekly Average", f"{stats['weekly_average']:.1f}")
        
        # Recent attendance
        st.subheader("Recent Attendance")
        recent_data = self.get_recent_attendance()
        if not recent_data.empty:
            st.dataframe(recent_data)

    def show_registration(self):
        """User registration page"""
        st.title("Register New User")
        
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                
            with col2:
                department = st.selectbox(
                    "Department",
                    ["IT", "HR", "Finance", "Operations", "Marketing"]
                )
                employee_id = st.text_input("Employee ID")
            
            uploaded_file = st.file_uploader(
                "Upload Photo", 
                type=['jpg', 'jpeg', 'png']
            )
            
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if all([full_name, email, department, employee_id, uploaded_file]):
                    success, message = self.register_user(
                        full_name, email, department, employee_id, uploaded_file
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill all fields and upload a photo")

    def show_attendance(self):
        """Attendance capture page"""
        st.title("Take Attendance")
        
        # Get known face data
        known_face_data = self.get_user_data()
        if not known_face_data:
            st.warning("No registered users found")
            return
            
        known_face_encodings = [data[2] for data in known_face_data]
        
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_button = st.button("Stop")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break
                
                # Process frame
                face_locations, face_names = self.process_face_recognition(
                    frame, known_face_encodings, known_face_data
                )
                
                # Draw results
                for (top, right, bottom, left), (name, confidence) in zip(
                    face_locations, face_names
                ):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw box and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, -1)
                    cv2.putText(
                        frame,
                        f"{name} ({confidence:.2%})",
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                
                # Display frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame)
                
            cap.release()

    def show_reports(self):
        """Reports and analytics page"""
        st.title("Attendance Reports")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        if start_date and end_date:
            if start_date <= end_date:
                # Get attendance data
                df = self.get_attendance_report(start_date, end_date)
                
                if not df.empty:
                    # Show attendance trend
                    st.subheader("Attendance Trend")
                    fig = px.line(
                        df, 
                        x='date', 
                        y='attendance_count',
                        title='Daily Attendance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed report
                    st.subheader("Detailed Report")
                    st.dataframe(df)
                    
                    # Download option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Report",
                        csv,
                        "attendance_report.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.info("No attendance data found for selected date range")
            else:
                st.error("End date must be after start date")

def main():
    system = AttendanceSystem()
    system.streamlit_interface()

if __name__ == "__main__":
    main()
