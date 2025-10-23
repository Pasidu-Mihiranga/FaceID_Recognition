// Face ID System JavaScript

// Global variables
let isProcessing = false;
let recognitionHistory = [];

// Utility functions
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'times-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Image processing utilities
function resizeImage(file, maxWidth = 800, maxHeight = 600, quality = 0.8) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            let { width, height } = img;
            
            // Calculate new dimensions
            if (width > height) {
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }
            } else {
                if (height > maxHeight) {
                    width = (width * maxHeight) / height;
                    height = maxHeight;
                }
            }
            
            canvas.width = width;
            canvas.height = height;
            
            ctx.drawImage(img, 0, 0, width, height);
            
            canvas.toBlob(resolve, 'image/jpeg', quality);
        };
        
        img.src = URL.createObjectURL(file);
    });
}

function validateImageFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        throw new Error('Invalid file type. Please upload a JPG, PNG, GIF, or BMP image.');
    }
    
    if (file.size > maxSize) {
        throw new Error('File size too large. Please upload an image smaller than 16MB.');
    }
    
    return true;
}

// API functions
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

async function registerPerson(name, imageFile) {
    const formData = new FormData();
    formData.append('person_name', name);
    formData.append('file', imageFile);
    
    const response = await fetch('/api/register', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

async function recognizeFace(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch('/api/recognize', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

async function recognizeFaceBase64(imageData) {
    return await apiRequest('/api/recognize_base64', {
        method: 'POST',
        body: JSON.stringify({ image: imageData })
    });
}

async function getSystemStats() {
    return await apiRequest('/api/stats');
}

async function getPersons() {
    return await apiRequest('/api/persons');
}

async function deletePerson(personId) {
    return await apiRequest(`/api/delete_person/${personId}`, {
        method: 'DELETE'
    });
}

async function startCamera(cameraIndex = 0) {
    return await apiRequest('/api/camera/start', {
        method: 'POST',
        body: JSON.stringify({ camera_index: cameraIndex })
    });
}

async function stopCamera() {
    return await apiRequest('/api/camera/stop', {
        method: 'POST'
    });
}

async function exportData(exportPath = 'data/export') {
    return await apiRequest('/api/export', {
        method: 'POST',
        body: JSON.stringify({ export_path: exportPath })
    });
}

// Camera utilities
class CameraManager {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.context = null;
        this.isRunning = false;
        this.recognitionInterval = null;
    }
    
    async start(videoElement) {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });
            
            this.video = videoElement;
            this.video.srcObject = this.stream;
            this.video.play();
            
            this.canvas = document.createElement('canvas');
            this.context = this.canvas.getContext('2d');
            
            this.isRunning = true;
            
            // Start periodic recognition
            this.startPeriodicRecognition();
            
            return true;
        } catch (error) {
            console.error('Camera start failed:', error);
            throw error;
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.recognitionInterval) {
            clearInterval(this.recognitionInterval);
            this.recognitionInterval = null;
        }
        
        this.isRunning = false;
    }
    
    startPeriodicRecognition() {
        this.recognitionInterval = setInterval(() => {
            if (this.isRunning && this.video.videoWidth > 0) {
                this.captureAndRecognize();
            }
        }, 3000); // Recognize every 3 seconds
    }
    
    async captureAndRecognize() {
        try {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.context.drawImage(this.video, 0, 0);
            
            const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
            const result = await recognizeFaceBase64(imageData);
            
            if (result.person_name || result.confidence > 0.5) {
                this.onRecognitionResult(result);
            }
        } catch (error) {
            console.error('Recognition failed:', error);
        }
    }
    
    onRecognitionResult(result) {
        // Override this method in subclasses
        console.log('Recognition result:', result);
    }
}

// Chart utilities
class ChartManager {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.chart = null;
    }
    
    createDoughnutChart(data, options = {}) {
        if (this.chart) {
            this.chart.destroy();
        }
        
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        };
        
        this.chart = new Chart(this.canvas, {
            type: 'doughnut',
            data: data,
            options: { ...defaultOptions, ...options }
        });
    }
    
    updateData(newData) {
        if (this.chart) {
            this.chart.data = newData;
            this.chart.update();
        }
    }
    
    destroy() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }
}

// Form validation
class FormValidator {
    static validateRegistrationForm(name, file) {
        const errors = [];
        
        if (!name || name.trim().length < 2) {
            errors.push('Name must be at least 2 characters long');
        }
        
        if (!file) {
            errors.push('Please select an image file');
        } else {
            try {
                validateImageFile(file);
            } catch (error) {
                errors.push(error.message);
            }
        }
        
        return errors;
    }
    
    static validateRecognitionForm(file) {
        const errors = [];
        
        if (!file) {
            errors.push('Please select an image file');
        } else {
            try {
                validateImageFile(file);
            } catch (error) {
                errors.push(error.message);
            }
        }
        
        return errors;
    }
}

// Local storage utilities
class StorageManager {
    static set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
        }
    }
    
    static get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Failed to read from localStorage:', error);
            return defaultValue;
        }
    }
    
    static remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('Failed to remove from localStorage:', error);
        }
    }
    
    static clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.error('Failed to clear localStorage:', error);
        }
    }
}

// Event handlers
function handleImagePreview(input, previewElement) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewElement.innerHTML = `
                <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 200px;" alt="Preview">
                <p class="mt-2 text-muted small">${file.name}</p>
            `;
        };
        reader.readAsDataURL(file);
    }
}

function handleFormSubmission(form, submitCallback) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (isProcessing) {
            return;
        }
        
        isProcessing = true;
        
        try {
            await submitCallback();
        } catch (error) {
            console.error('Form submission error:', error);
            showNotification('An error occurred. Please try again.', 'danger');
        } finally {
            isProcessing = false;
        }
    });
}

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add loading states to buttons
    document.querySelectorAll('button[type="submit"]').forEach(button => {
        button.addEventListener('click', function() {
            if (!isProcessing) {
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                this.disabled = true;
                
                setTimeout(() => {
                    this.innerHTML = originalText;
                    this.disabled = false;
                }, 2000);
            }
        });
    });
});

// Export utilities for use in other scripts
window.FaceIDUtils = {
    showNotification,
    formatDate,
    formatConfidence,
    debounce,
    resizeImage,
    validateImageFile,
    apiRequest,
    registerPerson,
    recognizeFace,
    recognizeFaceBase64,
    getSystemStats,
    getPersons,
    deletePerson,
    startCamera,
    stopCamera,
    exportData,
    CameraManager,
    ChartManager,
    FormValidator,
    StorageManager,
    handleImagePreview,
    handleFormSubmission
};
