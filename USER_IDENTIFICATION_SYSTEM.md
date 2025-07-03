# User Identification System

## Overview

The Audio Watermark Lab now includes a comprehensive user identification system that automatically tracks and displays user information throughout the watermarking process.

## How It Works

### 1. User Registration
- When users log in or register, their information is automatically stored in a local user registry
- User data includes: ID, name, role, and email
- This data persists across browser sessions using localStorage

### 2. Automatic User Detection
- The system no longer requires manual user selection
- The authenticated user's information is automatically used for watermarking
- User IDs are embedded in watermarks and can be traced back to specific users

### 3. User Name Resolution
- When detecting watermarks, the system automatically resolves user IDs to display names
- Instead of showing "User ID 1", it shows "Mayi (ID: 1)"
- This works for both known and unknown users

## Key Features

### Dashboard
- Shows current authenticated user information
- Automatically uses the logged-in user for watermarking
- Displays user names instead of just IDs in detection results
- Includes debug section showing all registered users

### Detect Page
- Automatically resolves user IDs to names in detection results
- Shows format: "Mayi (ID: 1)" instead of just "User ID: 1"

### Embed Page
- Shows current user information
- Uses authenticated user automatically

## Technical Implementation

### User Registry Service (`src/services/userRegistry.ts`)
- Singleton pattern for global user management
- localStorage persistence
- Automatic mock user initialization for testing
- Methods for user registration, retrieval, and name resolution

### AuthContext Integration
- Automatically registers users when they log in/register
- Generates consistent user IDs based on email hash
- Integrates with existing authentication flow

### Mock Users (for testing)
- Mayi (ID: 1) - Voice Actor
- Alice (ID: 2) - Producer  
- Bob (ID: 3) - Editor
- Charlie (ID: 4) - Marketer
- Diana (ID: 5) - Auditor

## Benefits

1. **Automatic User Tracking**: No manual user selection required
2. **Persistent User Data**: User information persists across sessions
3. **Human-Readable Names**: Detection results show actual names instead of IDs
4. **Consistent User IDs**: Same email always generates same user ID
5. **Easy Testing**: Mock users available for testing different scenarios

## Usage

1. Log in with any email/password combination
2. The system automatically creates/retrieves your user profile
3. Your user information is displayed in the interface
4. When embedding watermarks, your user ID is automatically included
5. When detecting watermarks, user names are automatically resolved and displayed

## Future Enhancements

- Database integration for persistent user storage
- User profile management
- Role-based permissions
- User activity tracking
- Advanced user analytics 