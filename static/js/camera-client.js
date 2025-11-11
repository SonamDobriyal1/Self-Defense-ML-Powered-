(function() {
    const socket = io();
    // Expose the socket so inline scripts can reuse the same connection
    window.socket = socket;
    let videoElement = null;
    let canvas = null;
    let streaming = false;
    let mediaStream = null;
    let streamInterval = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const FPS = 3; // Reduced FPS to lighten processing load
    const ENABLE_SIMILARITY_OVERLAY = false; // Disable top-left similarity overlay (shown below instead)

    // Create a lightweight on-screen overlay for similarity (disabled)
    let similarityOverlay = null;
    if (ENABLE_SIMILARITY_OVERLAY) {
        similarityOverlay = document.createElement('div');
        similarityOverlay.id = 'similarity-overlay';
        similarityOverlay.style.cssText = [
            'position:fixed',
            'left:16px',
            'bottom:16px',
            'z-index:9999',
            'padding:8px 12px',
            'border-radius:10px',
            'background:rgba(0,0,0,0.6)',
            'color:#fff',
            'font:600 14px/1.2 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
            'box-shadow:0 4px 12px rgba(0,0,0,0.3)',
            'backdrop-filter: blur(6px)',
            'display:none'
        ].join(';');
        similarityOverlay.textContent = 'Similarity: 0%';
        document.addEventListener('DOMContentLoaded', () => {
            document.body.appendChild(similarityOverlay);
        });
    }

    function setupCamera(videoId) {
        videoElement = document.getElementById(videoId);
        if (!videoElement) {
            console.error('Video element not found:', videoId);
            return false;
        }

        // Create a canvas element for capturing frames
        canvas = document.createElement('canvas');
        canvas.width = 480;
        canvas.height = 360;
        
        return true;
    }

    function startCamera(videoId) {
        if (!setupCamera(videoId)) return;

        console.log('Requesting camera access...');
        
        // Request access to the webcam
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: FPS }
            }, 
            audio: false 
        })
        .then(function(stream) {
            console.log('Camera access granted');
            mediaStream = stream;
            videoElement.srcObject = stream;
            videoElement.style.display = 'block'; // Show the video element
            videoElement.play();
            
            videoElement.onloadedmetadata = function(e) {
                console.log('Video metadata loaded, starting stream');
                streaming = true;
                startStreaming();
            };
        })
        .catch(function(err) {
            console.error("Camera access error: " + err);
            alert("Camera access is required for pose detection. Please allow camera access and refresh the page.");
        });
    }

    function stopCamera() {
        console.log('Stopping camera...');
        
        if (streamInterval) {
            clearInterval(streamInterval);
            streamInterval = null;
        }
        
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => {
                track.stop();
            });
            mediaStream = null;
        }
        
        if (videoElement) {
            videoElement.style.display = 'none';
        }
        
        streaming = false;
    }

    function captureFrame() {
        if (!streaming || !videoElement || videoElement.videoWidth === 0) {
            return null;
        }
        
        const context = canvas.getContext('2d');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        // Draw the current video frame to the canvas
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to jpeg data URL with reduced quality
        return canvas.toDataURL('image/jpeg', 0.5);
    }

    function startStreaming() {
        if (streamInterval) clearInterval(streamInterval);
        
        console.log('Starting frame streaming at', FPS, 'FPS...');
        
        // Send frames at regular intervals
        streamInterval = setInterval(() => {
            const frameData = captureFrame();
            if (frameData) {
                console.log('Sending frame to server, size:', frameData.length, 'characters');
                // Send frame data in the format the server expects
                socket.emit('frame', { image: frameData });
            } else {
                console.warn('Failed to capture frame');
            }
        }, 1000 / FPS);
    }

    const similarityLabelEl = document.getElementById('live-similarity');
    const videoPlaceholderEl = document.getElementById('video-placeholder');

    // Handle processed frames coming back from the server
    socket.on('processed_frame', function(data) {
        console.log('Received processed frame from server');
        const displayEl = document.getElementById('video-feed');
        if (displayEl) {
            displayEl.src = data.image;
            displayEl.style.display = 'block';
            console.log('Updated video-feed element with processed frame');
            if (videoPlaceholderEl) {
                videoPlaceholderEl.style.display = 'none';
            }
        } else {
            console.error('video-feed element not found');
        }
    });

    // Forward pose events to the page (so inline scripts can react)
    socket.on('pose_match_confirmed', function(data) {
        try {
            // If the page defined handlers, call them
            if (typeof window.showPoseSuccessMessage === 'function') {
                window.showPoseSuccessMessage(data);
            }
            // Auto-advance using page helper if available
            setTimeout(() => {
                if (typeof window.playNextVideo === 'function') {
                    // Update global index if present; fallback to local tracking
                    if (typeof window.currentVideoIndex === 'number') {
                        window.currentVideoIndex += 1;
                    }
                    window.playNextVideo();
                }
            }, 3000);
        } catch (e) {
            console.error('Error handling pose_match_confirmed in camera client', e);
        }
        // Also dispatch a DOM event for any other listeners
        try {
            const evt = new CustomEvent('pose_match_confirmed', { detail: data });
            window.dispatchEvent(evt);
        } catch {}
    });

    // Show live similarity updates (UI below handles display; skip overlay when disabled)
    socket.on('pose_similarity', function(data) {
        try {
            const val = typeof data?.similarity === 'number' ? data.similarity : 0;
            const pct = Math.max(0, Math.min(100, Math.round(val)));

            if (similarityLabelEl) {
                similarityLabelEl.textContent = `Similarity: ${pct}%`;
            }

            if (ENABLE_SIMILARITY_OVERLAY && similarityOverlay) {
                similarityOverlay.textContent = `Similarity: ${pct}%`;
                let border = '#dc3545';
                if (pct >= 85) border = '#28a745';
                else if (pct >= 75) border = '#ffc107';
                similarityOverlay.style.border = `1px solid ${border}`;
                similarityOverlay.style.display = 'block';
            }

            const customEvent = new CustomEvent('pose_similarity', { detail: data });
            window.dispatchEvent(customEvent);
        } catch (e) {
            console.error('Error updating similarity overlay', e);
        }
    });

    // Handle socket connection events
    socket.on('connect', function() {
        console.log('Socket connected successfully');
        reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        
        // Restart streaming if camera was active
        if (streaming && videoElement) {
            console.log('Reconnected - restarting frame streaming...');
            startStreaming();
        }
    });

    socket.on('disconnect', function(reason) {
        console.log('Socket disconnected - reason:', reason);
        // Stop camera streaming when disconnected
        if (streamInterval) {
            clearInterval(streamInterval);
            streamInterval = null;
            console.log('Stopped frame streaming due to disconnection');
        }
        
        // Attempt to reconnect if it's not an intentional disconnect
        if (reason !== 'io client disconnect' && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect... (${reconnectAttempts}/${maxReconnectAttempts})`);
            setTimeout(() => {
                socket.connect();
            }, 2000 * reconnectAttempts); // Exponential backoff
        }
    });

    socket.on('connect_error', function(error) {
        console.error('Socket connection error:', error);
        reconnectAttempts++;
        if (reconnectAttempts < maxReconnectAttempts) {
            console.log(`Retrying connection in ${2000 * reconnectAttempts}ms...`);
        }
    });

    socket.on('error', function(error) {
        console.error('Socket error:', error);
    });

    // Expose functions to global scope
    window.cameraClient = {
        start: startCamera,
        stop: stopCamera
    };
})();
