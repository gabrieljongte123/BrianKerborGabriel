class DigitRecognitionApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.initializeCanvas();
        this.bindEvents();
    }

    initializeCanvas() {
        // Set canvas background to white
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Set drawing style
        this.ctx.strokeStyle = 'black';
        this.ctx.lineWidth = 15;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    bindEvents() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        // Button events
        document.getElementById('clearBtn').addEventListener('click', this.clearCanvas.bind(this));
        document.getElementById('predictBtn').addEventListener('click', this.predictDigit.bind(this));
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    getTouchPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const touch = e.touches[0];
        return {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.lastX = pos.x;
        this.lastY = pos.y;
    }

    draw(e) {
        if (!this.isDrawing) return;
        e.preventDefault();

        const pos = this.getMousePos(e);
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();

        this.lastX = pos.x;
        this.lastY = pos.y;
    }

    handleTouch(e) {
        e.preventDefault();
        const pos = this.getTouchPos(e);
        
        if (e.type === 'touchstart') {
            this.isDrawing = true;
            this.lastX = pos.x;
            this.lastY = pos.y;
        } else if (e.type === 'touchmove' && this.isDrawing) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.lastX, this.lastY);
            this.ctx.lineTo(pos.x, pos.y);
            this.ctx.stroke();
            
            this.lastX = pos.x;
            this.lastY = pos.y;
        }
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    clearCanvas() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.hideResults();
    }

    hideResults() {
        document.getElementById('result').classList.add('hidden');
        document.getElementById('error').classList.add('hidden');
        document.getElementById('loading').classList.add('hidden');
    }

    showLoading() {
        this.hideResults();
        document.getElementById('loading').classList.remove('hidden');
    }

    showError(message) {
        this.hideResults();
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('error').classList.remove('hidden');
    }

    showResults(data) {
        this.hideResults();
        
        // Update prediction
        document.getElementById('predictedDigit').textContent = data.prediction;
        
        // Update confidence
        const confidencePercent = Math.round(data.confidence * 100);
        document.getElementById('confidenceText').textContent = `${confidencePercent}%`;
        document.getElementById('confidenceFill').style.width = `${confidencePercent}%`;
        
        // Update probability bars
        this.updateProbabilityBars(data.probabilities);
        
        document.getElementById('result').classList.remove('hidden');
    }

    updateProbabilityBars(probabilities) {
        const container = document.getElementById('probabilityBars');
        container.innerHTML = '';
        
        probabilities.forEach((prob, digit) => {
            const percent = Math.round(prob * 100);
            const item = document.createElement('div');
            item.className = 'probability-item';
            item.innerHTML = `
                <div class="probability-digit">${digit}</div>
                <div class="probability-value">${percent}%</div>
            `;
            container.appendChild(item);
        });
    }

    async predictDigit() {
        // Check if canvas has any drawing
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const hasDrawing = imageData.data.some(pixel => pixel < 255);
        
        if (!hasDrawing) {
            this.showError('Please draw a digit first!');
            return;
        }

        this.showLoading();

        try {
            // Convert canvas to base64
            const imageData = this.canvas.toDataURL('image/png');
            
            // Send to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            this.showResults(data);

        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(error.message || 'Failed to predict digit. Please try again.');
        }
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new DigitRecognitionApp();
}); 