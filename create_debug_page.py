#!/usr/bin/env python3
"""
Create a simple HTML test page to debug form submission
"""

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Face ID Registration Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .form-group { margin: 10px 0; }
        input, button { padding: 10px; margin: 5px; }
        .success { color: green; }
        .error { color: red; }
        .debug { background: #f0f0f0; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Face ID Registration Test</h1>
    
    <div class="debug">
        <h3>Debug Information:</h3>
        <div id="debug-info">Loading...</div>
    </div>
    
    <form id="test-form" enctype="multipart/form-data">
        <div class="form-group">
            <label>Person Name:</label><br>
            <input type="text" id="person_name" name="person_name" value="Test Person" required>
        </div>
        
        <div class="form-group">
            <label>Image File:</label><br>
            <input type="file" id="image_file" name="file" accept="image/*" required>
        </div>
        
        <div class="form-group">
            <button type="submit">Register Person</button>
            <button type="button" onclick="testForm()">Test Form</button>
            <button type="button" onclick="checkServer()">Check Server</button>
        </div>
    </form>
    
    <div id="result"></div>
    
    <script>
        // Debug information
        function updateDebug() {
            const debugDiv = document.getElementById('debug-info');
            debugDiv.innerHTML = `
                <strong>Current Status:</strong><br>
                Server: <span id="server-status">Checking...</span><br>
                Form Elements: <span id="form-status">Checking...</span><br>
                JavaScript: <span id="js-status">Working</span><br>
                Console: <span id="console-status">Open F12 to see logs</span>
            `;
        }
        
        // Check server status
        function checkServer() {
            fetch('/api/persons')
                .then(response => {
                    document.getElementById('server-status').innerHTML = 
                        response.ok ? '<span class="success">Connected</span>' : '<span class="error">Error</span>';
                })
                .catch(error => {
                    document.getElementById('server-status').innerHTML = '<span class="error">Not Connected</span>';
                });
        }
        
        // Test form validation
        function testForm() {
            const personName = document.getElementById('person_name').value;
            const imageFile = document.getElementById('image_file').files[0];
            
            let status = 'Form validation: ';
            if (!personName) {
                status += '<span class="error">Missing name</span>';
            } else if (!imageFile) {
                status += '<span class="error">Missing image</span>';
            } else {
                status += '<span class="success">Valid</span>';
            }
            
            document.getElementById('form-status').innerHTML = status;
            document.getElementById('result').innerHTML = status;
        }
        
        // Form submission
        document.getElementById('test-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            console.log('=== FORM SUBMISSION DEBUG ===');
            
            const personName = document.getElementById('person_name').value;
            const imageFile = document.getElementById('image_file').files[0];
            
            console.log('Person name:', personName);
            console.log('Image file:', imageFile);
            
            if (!personName || !imageFile) {
                console.log('Validation failed');
                document.getElementById('result').innerHTML = '<div class="error">Please fill in all fields</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('person_name', personName);
            formData.append('file', imageFile);
            
            console.log('FormData prepared, sending request...');
            document.getElementById('result').innerHTML = '<div>Submitting...</div>';
            
            fetch('/api/register', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Response data:', data);
                if (data.success) {
                    document.getElementById('result').innerHTML = 
                        '<div class="success">SUCCESS: ' + data.message + '</div>';
                } else {
                    document.getElementById('result').innerHTML = 
                        '<div class="error">ERROR: ' + (data.error || 'Registration failed') + '</div>';
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
                document.getElementById('result').innerHTML = 
                    '<div class="error">ERROR: ' + error.message + '</div>';
            });
        });
        
        // Initialize
        updateDebug();
        checkServer();
        testForm();
        
        // Update debug info every 5 seconds
        setInterval(() => {
            updateDebug();
            checkServer();
        }, 5000);
    </script>
</body>
</html>
"""

with open('debug_registration.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Created debug_registration.html")
print("Open this file in your browser to test the registration form")
print("URL: http://localhost:5000/debug_registration.html")
