import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

class InnovationLabCounter {
    constructor() {
        this.video = document.getElementById('video');
        this.overlay = document.getElementById('overlay');
        this.cameraSelect = document.getElementById('cameraSelect');
        this.startButton = document.getElementById('startButton');
        this.roiButton = document.getElementById('roiButton');
        this.countDisplay = document.getElementById('count');
        this.settingsToggle = document.querySelector('.settings-toggle');
        this.settingsPanel = document.querySelector('.settings-panel');
        
        this.isRunning = false;
        this.roiPoints = [];
        this.drawingROI = false;
        this.model = null;
        this.ctx = this.overlay.getContext('2d');
        
        this.init();
    }

    async init() {
        try {
            await tf.ready();
            console.log('TensorFlow.js is ready');
            
            console.log('Loading COCO-SSD model...');
            this.model = await cocoSsd.load({
                base: 'mobilenet_v2'
            });
            console.log('Model loaded successfully');

            this.setupEventListeners();
            await this.loadCameras();
        } catch (error) {
            console.error('Error initializing:', error);
        }
    }

    setupEventListeners() {
        this.startButton.addEventListener('click', () => this.toggleCamera());
        this.roiButton.addEventListener('click', () => this.toggleROI());
        this.overlay.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.settingsToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            this.settingsPanel.classList.toggle('active');
        });

        document.addEventListener('click', (e) => {
            if (!e.target.closest('.settings-panel') && !e.target.closest('.settings-toggle')) {
                this.settingsPanel.classList.remove('active');
            }
        });

        this.video.addEventListener('loadedmetadata', () => {
            this.overlay.width = this.video.videoWidth;
            this.overlay.height = this.video.videoHeight;
        });
    }

    async loadCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            this.cameraSelect.innerHTML = '<option value="">Select Camera</option>';
            videoDevices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${this.cameraSelect.length}`;
                this.cameraSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading cameras:', error);
        }
    }

    async toggleCamera() {
        if (this.isRunning) {
            this.stopCamera();
        } else {
            await this.startCamera();
        }
    }

    async startCamera() {
        try {
            const constraints = {
                video: {
                    deviceId: this.cameraSelect.value ? { exact: this.cameraSelect.value } : undefined,
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = stream;
            
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });

            this.isRunning = true;
            this.startButton.textContent = 'Stop Camera';
            this.detectPeople();
        } catch (error) {
            console.error('Error starting camera:', error);
        }
    }

    stopCamera() {
        if (this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
        }
        this.isRunning = false;
        this.startButton.textContent = 'Start Camera';
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        this.countDisplay.textContent = '0';
    }

    toggleROI() {
        this.drawingROI = !this.drawingROI;
        this.roiButton.textContent = this.drawingROI ? 'Drawing ROI...' : 'Set ROI';
        if (this.drawingROI) {
            this.roiPoints = [];
            this.drawROI();
        }
    }

    handleCanvasClick(event) {
        if (!this.drawingROI) return;

        const rect = this.overlay.getBoundingClientRect();
        const scaleX = this.overlay.width / rect.width;
        const scaleY = this.overlay.height / rect.height;
        
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;
        
        this.roiPoints.push({ x, y });
        this.drawROI();
        
        if (this.roiPoints.length === 4) {
            this.drawingROI = false;
            this.roiButton.textContent = 'Set ROI';
        }
    }

    drawROI() {
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        
        if (this.roiPoints.length < 2) return;
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.roiPoints[0].x, this.roiPoints[0].y);
        
        for (let i = 1; i < this.roiPoints.length; i++) {
            this.ctx.lineTo(this.roiPoints[i].x, this.roiPoints[i].y);
        }
        
        if (this.roiPoints.length === 4) {
            this.ctx.closePath();
        }
        
        this.ctx.strokeStyle = '#00ff66';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
    }

    async detectPeople() {
        if (!this.isRunning || !this.model || !this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
            requestAnimationFrame(() => this.detectPeople());
            return;
        }

        try {
            const predictions = await this.model.detect(this.video);
            
            this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
            this.drawROI();
            
            const people = predictions.filter(prediction => prediction.class === 'person');
            const validPeople = people.filter(person => !this.roiPoints.length || this.isInROI(person.bbox));
            const count = validPeople.length;
            
            // Update counter with proper pluralization
            const counterText = count === 1 ? 
                "1 person currently enjoying the lab" : 
                `${count} people currently enjoying the lab`;
            
            document.querySelector('.counter').textContent = counterText;
            
            validPeople.forEach(person => {
                const [x, y, width, height] = person.bbox;
                
                // Draw box with glow effect
                this.ctx.shadowColor = '#00ff66';
                this.ctx.shadowBlur = 15;
                this.ctx.strokeStyle = '#00ff66';
                this.ctx.lineWidth = 3;
                this.ctx.strokeRect(x, y, width, height);
                
                // Reset shadow for text
                this.ctx.shadowBlur = 0;
                
                // Draw confidence score
                const confidence = Math.round(person.score * 100);
                this.ctx.fillStyle = '#00ff66';
                this.ctx.font = 'bold 24px Arial';
                this.ctx.fillText(`${confidence}%`, x, y - 10);
            });
        } catch (error) {
            console.error('Error in person detection:', error);
        }

        requestAnimationFrame(() => this.detectPeople());
    }

    isInROI(bbox) {
        if (this.roiPoints.length !== 4) return true;
        
        const [x, y, width, height] = bbox;
        const center = {
            x: x + width / 2,
            y: y + height / 2
        };

        let inside = false;
        for (let i = 0, j = this.roiPoints.length - 1; i < this.roiPoints.length; j = i++) {
            const xi = this.roiPoints[i].x;
            const yi = this.roiPoints[i].y;
            const xj = this.roiPoints[j].x;
            const yj = this.roiPoints[j].y;

            const intersect = ((yi > center.y) !== (yj > center.y))
                && (center.x < (xj - xi) * (center.y - yi) / (yj - yi) + xi);
            
            if (intersect) inside = !inside;
        }
        
        return inside;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    new InnovationLabCounter();
});