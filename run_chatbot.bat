@echo off
echo تنشيط البيئة الافتراضية...
call .\venv\Scripts\activate.bat

echo تشغيل تطبيق الشات بوت...
python app.py

pause