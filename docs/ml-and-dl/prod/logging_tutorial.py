"""
=============================================================================
                    PYTHON LOGGING - BASIC TUTORIAL
=============================================================================

Author: Based on Python's official Logging HOWTO
Purpose: Learn the fundamentals of Python's logging module

WHAT IS LOGGING?
    Logging is a way to track events that happen when your software runs.
    Instead of using print() everywhere, logging gives you:
    - Different severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Easy control over what gets shown/hidden
    - Ability to log to files, console, or both
    - Timestamps and structured output

WHEN TO USE LOGGING vs PRINT:
    - print()           → Display output for normal program usage
    - logging.info()    → Report normal events during operation
    - logging.debug()   → Detailed diagnostic info for debugging
    - logging.warning() → Something unexpected, but program still works
    - logging.error()   → Serious problem, some function failed
    - logging.critical()→ Program may crash or cannot continue

LOGGING LEVELS (in order of severity):
    Level       Value   When to use
    ─────────────────────────────────────────────────────────────────
    DEBUG       10      Detailed info for diagnosing problems
    INFO        20      Confirmation that things are working
    WARNING     30      Something unexpected happened (DEFAULT level)
    ERROR       40      A function failed due to a problem
    CRITICAL    50      Program itself may be unable to continue
"""

import logging

# =============================================================================
# EXAMPLE 1: Simple Logging (Root Logger)
# =============================================================================
# The simplest way to log - uses the "root" logger
# Default level is WARNING, so DEBUG and INFO won't show!

print("\n" + "="*60)
print("EXAMPLE 1: Simple Logging (Default WARNING level)")
print("="*60)

logging.warning('Watch out!')       # ✓ Will print
logging.info('I told you so')       # ✗ Won't print (below WARNING threshold)

# Output: WARNING:root:Watch out!


# =============================================================================
# EXAMPLE 2: Configure Logging Level
# =============================================================================
# Use basicConfig() to change settings BEFORE any logging calls
# Here we set level to DEBUG so ALL messages show

print("\n" + "="*60)
print("EXAMPLE 2: Setting the Logging Level to DEBUG")
print("="*60)

# NOTE: basicConfig() only works ONCE. For this demo, we use force=True
logging.basicConfig(level=logging.DEBUG, force=True)

logging.debug('This is a DEBUG message')       # Detailed diagnostic info
logging.info('This is an INFO message')        # Confirmation of normal operation
logging.warning('This is a WARNING message')   # Something unexpected
logging.error('This is an ERROR message')      # Something failed
logging.critical('This is a CRITICAL message') # Program may crash


# =============================================================================
# EXAMPLE 3: Using a Named Logger (RECOMMENDED)
# =============================================================================
# For real projects, create a named logger instead of using the root logger
# This gives you better control and shows which module the log came from

print("\n" + "="*60)
print("EXAMPLE 3: Using a Named Logger")
print("="*60)

# Create a logger with the module's name
logger = logging.getLogger(__name__)

logger.debug('Debug from named logger')
logger.info('Info from named logger')
logger.warning('Warning from named logger')

# Output shows __main__ instead of root:
# DEBUG:__main__:Debug from named logger


# =============================================================================
# EXAMPLE 4: Custom Format
# =============================================================================
# Customize what info appears in log messages

print("\n" + "="*60)
print("EXAMPLE 4: Custom Message Format")
print("="*60)

# Available format attributes:
#   %(levelname)s  - Level name (DEBUG, INFO, etc.)
#   %(message)s    - The log message itself
#   %(asctime)s    - Timestamp when the log was created
#   %(name)s       - Logger name
#   %(filename)s   - Source filename
#   %(lineno)d     - Line number in source code

logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    force=True
)

logging.info('This message has a nice timestamp')
logging.warning('Warnings are also formatted')

# Output: 2024-01-20 22:30:00 | INFO     | This message has a nice timestamp


# =============================================================================
# EXAMPLE 5: Logging Variable Data
# =============================================================================
# Use %-style formatting to include variables in log messages

print("\n" + "="*60)
print("EXAMPLE 5: Logging with Variables")
print("="*60)

user_name = "Alice"
user_age = 28
items_count = 42

# Method 1: %-style formatting (recommended for logging)
logging.info('User %s logged in', user_name)
logging.info('User age: %d, Items: %d', user_age, items_count)

# Method 2: f-strings (also works, but less efficient for logging)
logging.info(f'User {user_name} has {items_count} items')

# %-style is preferred because the string formatting only happens
# if the message will actually be logged (based on level)


# =============================================================================
# EXAMPLE 6: Logging to a File
# =============================================================================
# Save logs to a file instead of (or in addition to) the console

print("\n" + "="*60)
print("EXAMPLE 6: Logging to a File")
print("="*60)

# This creates/overwrites 'app.log' file
logging.basicConfig(
    filename='app.log',
    filemode='w',           # 'w' = overwrite, 'a' = append (default)
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    force=True
)

logging.debug('This goes to the file')
logging.info('So does this')
logging.warning('And this warning too')
logging.error('Errors are logged as well')

print("Check 'app.log' file for the logged messages!")


# =============================================================================
# QUICK REFERENCE
# =============================================================================
"""
QUICK REFERENCE CHEAT SHEET
───────────────────────────────────────────────────────────────────────────

1. BASIC SETUP:
   import logging
   logging.basicConfig(level=logging.DEBUG)

2. CREATE A NAMED LOGGER:
   logger = logging.getLogger(__name__)

3. LOG MESSAGES:
   logger.debug('Detailed debug info')
   logger.info('Normal operation info')
   logger.warning('Warning message')
   logger.error('Error occurred')
   logger.critical('Critical failure')

4. LOG TO FILE:
   logging.basicConfig(filename='app.log', level=logging.DEBUG)

5. CUSTOM FORMAT:
   logging.basicConfig(
       format='%(asctime)s - %(levelname)s - %(message)s',
       level=logging.DEBUG
   )

6. LOG WITH VARIABLES:
   logger.info('User %s logged in with %d items', username, count)

COMMON FORMAT SPECIFIERS:
   %(asctime)s   → Timestamp
   %(levelname)s → Level (DEBUG, INFO, etc.)
   %(message)s   → Your log message
   %(name)s      → Logger name
   %(filename)s  → Source file name
   %(lineno)d    → Line number
"""


# =============================================================================
# RUN THIS FILE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TUTORIAL COMPLETE!")
    print("="*60)
    print("""
Key Takeaways:
1. Use logging instead of print() for tracking events
2. Configure with basicConfig() before any logging calls
3. Use getLogger(__name__) for named loggers in real projects
4. Choose the right level: DEBUG < INFO < WARNING < ERROR < CRITICAL
5. Default level is WARNING - set lower to see more messages
""")
