import sys
import traceback
try:
    sys.path.insert(0, r"C:\Users\ph\Spam-message-detection-main\spam_app")
    import app as app_module
    print('Imported app module OK')
except Exception as e:
    traceback.print_exc()
