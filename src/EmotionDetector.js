import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, StopCircle, AlertCircle, Download, CheckCircle } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

const EmotionDetector = () => {
    const [mode, setMode] = useState('upload');
    const [isWebcamActive, setIsWebcamActive] = useState(false);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [emotions, setEmotions] = useState(null);
    const [model, setModel] = useState(null);
    const [modelLoading, setModelLoading] = useState(true);
    const [modelError, setModelError] = useState(null);
    const [tfReady, setTfReady] = useState(false);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const detectionIntervalRef = useRef(null);

    // Danh sách cảm xúc (theo thứ tự model FER2013)
    const EMOTIONS = ['Tức giận', 'Ghê tởm', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn', 'Ngạc nhiên'];

    const emotionColors = {
        'Hạnh phúc': 'bg-yellow-500',
        'Buồn': 'bg-blue-500',
        'Tức giận': 'bg-red-500',
        'Ngạc nhiên': 'bg-purple-500',
        'Sợ hãi': 'bg-gray-600',
        'Ghê tởm': 'bg-green-600',
        'Bình thường': 'bg-gray-400'
    };

    const emotionEmojis = {
        'Hạnh phúc': '😊',
        'Buồn': '😢',
        'Tức giận': '😠',
        'Ngạc nhiên': '😲',
        'Sợ hãi': '😨',
        'Ghê tởm': '🤢',
        'Bình thường': '😐'
    };

    // Load TensorFlow.js và model
    useEffect(() => {
        const loadModel = async () => {
            try {
                setModelLoading(true);

                // Kiểm tra TensorFlow.js đã sẵn sàng
                await tf.ready();
                setTfReady(true);
                console.log('✅ TensorFlow.js ready');
                console.log('Backend:', tf.getBackend());

                // ⚠️ QUAN TRỌNG: Thay URL này bằng URL model của bạn
                // Các tùy chọn host model:
                // 1. GitHub Pages: https://yourusername.github.io/your-repo/tfjs_model/model.json
                // 2. Firebase Storage: https://firebasestorage.googleapis.com/...
                // 3. Vercel/Netlify: https://your-domain.vercel.app/tfjs_model/model.json

                const MODEL_URL = './tfjs_model/model.json'; // Local (sau khi copy vào public/)
                // const MODEL_URL = 'https://yourusername.github.io/emotion-model/model.json'; // GitHub

                // Uncomment 3 dòng dưới khi đã có model
                /*
                const loadedModel = await tf.loadLayersModel(MODEL_URL);
                setModel(loadedModel);
                console.log('✅ Model loaded successfully');
                */

                // Hiện tại dùng demo
                console.log('⚠️ Đang dùng mode demo. Vui lòng uncomment code load model khi đã có model.');
                setModelError('Chưa có model thực. Đang dùng demo với dữ liệu ngẫu nhiên.');

                setModelLoading(false);

            } catch (error) {
                console.error('❌ Error loading model:', error);
                setModelError(`Lỗi: ${error.message}`);
                setModelLoading(false);
                setTfReady(true);
            }
        };

        loadModel();

        return () => {
            if (model) {
                model.dispose();
            }
        };
    }, []);

    // Tiền xử lý ảnh cho model
    const preprocessImage = (imageElement) => {
        return tf.tidy(() => {
            // Chuyển ảnh sang tensor
            let tensor = tf.browser.fromPixels(imageElement, 1); // 1 = grayscale

            // Resize về 48x48 (kích thước model FER2013)
            tensor = tf.image.resizeBilinear(tensor, [48, 48]);

            // Chuẩn hóa [0, 255] -> [0, 1]
            tensor = tensor.div(255.0);

            // Thêm batch dimension [1, 48, 48, 1]
            tensor = tensor.expandDims(0);

            return tensor;
        });
    };

    // Phân tích cảm xúc với model thực
    const analyzeEmotionReal = async (imageElement) => {
        if (!model) {
            console.warn('Model chưa được load, dùng demo');
            return analyzeEmotionDemo();
        }

        try {
            setAnalyzing(true);

            // Tiền xử lý ảnh
            const tensor = preprocessImage(imageElement);

            // Dự đoán
            const predictions = model.predict(tensor);
            const probabilities = await predictions.data();

            // Chuyển thành object với tên cảm xúc
            const results = {};
            EMOTIONS.forEach((emotion, index) => {
                results[emotion] = Math.round(probabilities[index] * 100);
            });

            // Sắp xếp giảm dần
            const sortedResults = Object.entries(results)
                .sort(([, a], [, b]) => b - a)
                .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

            setEmotions(sortedResults);

            // Cleanup tensors
            tensor.dispose();
            predictions.dispose();

            console.log('Predictions:', sortedResults);

        } catch (error) {
            console.error('Error during prediction:', error);
            analyzeEmotionDemo();
        } finally {
            setAnalyzing(false);
        }
    };

    // Demo với dữ liệu ngẫu nhiên
    const analyzeEmotionDemo = async () => {
        setAnalyzing(true);

        await new Promise(resolve => setTimeout(resolve, 1200));

        const results = {};
        let remaining = 100;

        // Tạo phân phối ngẫu nhiên
        EMOTIONS.forEach((emotion, index) => {
            if (index === EMOTIONS.length - 1) {
                results[emotion] = Math.max(0, remaining);
            } else {
                const value = Math.floor(Math.random() * (remaining / 2));
                results[emotion] = value;
                remaining -= value;
            }
        });

        const sortedResults = Object.entries(results)
            .sort(([, a], [, b]) => b - a)
            .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

        setEmotions(sortedResults);
        setAnalyzing(false);
    };

    // Upload ảnh
    const handleImageUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            setUploadedImage(event.target.result);

            // Tạo image element
            const img = new Image();
            img.onload = async () => {
                if (model) {
                    await analyzeEmotionReal(img);
                } else {
                    await analyzeEmotionDemo();
                }
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    };

    // Khởi động webcam
    const startWebcam = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                }
            });

            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsWebcamActive(true);

                // Phát hiện mỗi 2 giây
                detectionIntervalRef.current = setInterval(() => {
                    captureAndAnalyze();
                }, 2000);
            }
        } catch (err) {
            console.error('Webcam error:', err);
            alert('Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập trong trình duyệt.');
        }
    };

    // Capture từ webcam và phân tích
    const captureAndAnalyze = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const video = videoRef.current;

        if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (model) {
            await analyzeEmotionReal(canvas);
        } else {
            await analyzeEmotionDemo();
        }
    };

    // Dừng webcam
    const stopWebcam = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (detectionIntervalRef.current) {
            clearInterval(detectionIntervalRef.current);
            detectionIntervalRef.current = null;
        }
        setIsWebcamActive(false);
        setEmotions(null);
    };

    // Cleanup
    useEffect(() => {
        return () => {
            stopWebcam();
        };
    }, []);

    // Render kết quả
    const renderEmotionResults = () => {
        if (!emotions) return null;

        const topEmotion = Object.entries(emotions)[0];
        const emoji = emotionEmojis[topEmotion[0]];

        return (
            <div className="mt-6 bg-white rounded-lg shadow-lg p-6 animate-fadeIn">
                <div className="text-center mb-6">
                    <div className="text-7xl mb-3">{emoji}</div>
                    <h3 className="text-2xl font-bold text-gray-800">Cảm xúc chính</h3>
                    <p className="text-5xl font-bold text-purple-600 mt-2">{topEmotion[0]}</p>
                    <p className="text-2xl text-gray-600 mt-1">{topEmotion[1]}%</p>
                </div>

                <div className="space-y-3">
                    {Object.entries(emotions).map(([emotion, value]) => (
                        <div key={emotion} className="transform transition-all hover:scale-105">
                            <div className="flex justify-between text-sm mb-1">
                                <span className="font-medium text-gray-700 flex items-center gap-2">
                                    <span>{emotionEmojis[emotion]}</span>
                                    <span>{emotion}</span>
                                </span>
                                <span className="text-gray-600 font-semibold">{value}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                                <div
                                    className={`h-3 rounded-full transition-all duration-1000 ease-out ${emotionColors[emotion]}`}
                                    style={{ width: `${value}%` }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 p-4 md:p-6">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-3">
                        🤖 AI Nhận diện Cảm xúc
                    </h1>
                    <p className="text-gray-600 text-lg">
                        Sử dụng Deep Learning với TensorFlow.js
                    </p>

                    {/* Status indicators */}
                    <div className="mt-4 flex flex-wrap gap-3 justify-center">
                        {tfReady && (
                            <div className="inline-flex items-center gap-2 bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm">
                                <CheckCircle className="w-4 h-4" />
                                <span>TensorFlow.js Ready</span>
                            </div>
                        )}

                        {modelLoading && (
                            <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm">
                                <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-700 border-t-transparent"></div>
                                <span>Đang tải model...</span>
                            </div>
                        )}

                        {modelError && !modelLoading && (
                            <div className="inline-flex items-center gap-2 bg-yellow-100 text-yellow-800 px-4 py-2 rounded-full text-sm max-w-md">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                <span className="text-left">{modelError}</span>
                            </div>
                        )}

                        {model && !modelLoading && (
                            <div className="inline-flex items-center gap-2 bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm">
                                <CheckCircle className="w-4 h-4" />
                                <span>Model đã load</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Mode selector */}
                <div className="flex gap-3 md:gap-4 mb-6 justify-center">
                    <button
                        onClick={() => {
                            setMode('upload');
                            stopWebcam();
                            setUploadedImage(null);
                            setEmotions(null);
                        }}
                        className={`flex items-center gap-2 px-4 md:px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'upload'
                            ? 'bg-purple-600 text-white shadow-lg scale-105'
                            : 'bg-white text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Upload className="w-5 h-5" />
                        <span className="hidden sm:inline">Upload Ảnh</span>
                        <span className="sm:hidden">Upload</span>
                    </button>
                    <button
                        onClick={() => {
                            setMode('webcam');
                            setUploadedImage(null);
                            setEmotions(null);
                        }}
                        className={`flex items-center gap-2 px-4 md:px-6 py-3 rounded-lg font-semibold transition-all ${mode === 'webcam'
                            ? 'bg-purple-600 text-white shadow-lg scale-105'
                            : 'bg-white text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        <Camera className="w-5 h-5" />
                        <span>Webcam</span>
                    </button>
                </div>

                {/* Main content */}
                <div className="bg-white rounded-xl shadow-2xl p-4 md:p-6">
                    {/* Upload Mode */}
                    {mode === 'upload' && (
                        <div>
                            <div className="border-4 border-dashed border-gray-300 rounded-lg p-8 md:p-12 text-center hover:border-purple-500 transition-all cursor-pointer">
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleImageUpload}
                                    className="hidden"
                                    id="imageUpload"
                                />
                                <label htmlFor="imageUpload" className="cursor-pointer">
                                    <Upload className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-4 text-gray-400" />
                                    <p className="text-lg md:text-xl font-semibold text-gray-700">
                                        Click để chọn ảnh
                                    </p>
                                    <p className="text-sm text-gray-500 mt-2">
                                        Hỗ trợ JPG, PNG, JPEG
                                    </p>
                                </label>
                            </div>

                            {uploadedImage && (
                                <div className="mt-6">
                                    <img
                                        src={uploadedImage}
                                        alt="Uploaded"
                                        className="max-w-full h-auto rounded-lg mx-auto shadow-lg"
                                        style={{ maxHeight: '500px' }}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Webcam Mode */}
                    {mode === 'webcam' && (
                        <div>
                            <div className="relative">
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full rounded-lg shadow-lg"
                                    style={{ display: isWebcamActive ? 'block' : 'none' }}
                                />
                                <canvas ref={canvasRef} className="hidden" />

                                {!isWebcamActive && (
                                    <div className="border-4 border-dashed border-gray-300 rounded-lg p-8 md:p-12 text-center">
                                        <Camera className="w-12 h-12 md:w-16 md:h-16 mx-auto mb-4 text-gray-400" />
                                        <p className="text-lg md:text-xl font-semibold text-gray-700 mb-2">
                                            Webcam chưa được bật
                                        </p>
                                        <p className="text-sm text-gray-500">
                                            Click nút bên dưới để bắt đầu
                                        </p>
                                    </div>
                                )}
                            </div>

                            <div className="mt-4 text-center">
                                {!isWebcamActive ? (
                                    <button
                                        onClick={startWebcam}
                                        className="bg-green-600 text-white px-6 md:px-8 py-3 rounded-lg font-semibold hover:bg-green-700 transition-all shadow-lg hover:shadow-xl flex items-center gap-2 mx-auto"
                                    >
                                        <Camera className="w-5 h-5" />
                                        Bật Webcam
                                    </button>
                                ) : (
                                    <button
                                        onClick={stopWebcam}
                                        className="bg-red-600 text-white px-6 md:px-8 py-3 rounded-lg font-semibold hover:bg-red-700 transition-all shadow-lg hover:shadow-xl flex items-center gap-2 mx-auto"
                                    >
                                        <StopCircle className="w-5 h-5" />
                                        Dừng Webcam
                                    </button>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Loading state */}
                    {analyzing && (
                        <div className="mt-6 text-center">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-purple-600 border-t-transparent"></div>
                            <p className="mt-3 text-gray-600 font-medium">Đang phân tích cảm xúc...</p>
                        </div>
                    )}

                    {/* Results */}
                    {!analyzing && renderEmotionResults()}
                </div>

                {/* Instructions */}
                <div className="mt-6 bg-white rounded-lg shadow-lg p-4 md:p-6">
                    <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                        <Download className="w-5 h-5 text-purple-600" />
                        Hướng dẫn tích hợp Model
                    </h3>
                    <div className="space-y-3 text-sm text-gray-700">
                        <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
                            <p className="font-semibold mb-1">📝 Bước 1: Huấn luyện Model</p>
                            <p>Chạy code Python trên Google Colab để train model với dataset FER2013</p>
                        </div>

                        <div className="bg-green-50 border-l-4 border-green-500 p-3 rounded">
                            <p className="font-semibold mb-1">📦 Bước 2: Tải Model</p>
                            <p>Download file <code className="bg-gray-100 px-2 py-1 rounded">tfjs_model.zip</code> từ Colab</p>
                        </div>

                        <div className="bg-purple-50 border-l-4 border-purple-500 p-3 rounded">
                            <p className="font-semibold mb-1">🚀 Bước 3: Deploy Model</p>
                            <p>Upload thư mục model lên GitHub Pages hoặc host riêng</p>
                        </div>

                        <div className="bg-orange-50 border-l-4 border-orange-500 p-3 rounded">
                            <p className="font-semibold mb-1">⚙️ Bước 4: Cập nhật Code</p>
                            <p>Thay <code className="bg-gray-100 px-2 py-1 rounded">MODEL_URL</code> và uncomment code load model</p>
                        </div>
                    </div>

                    <div className="mt-4 text-xs text-gray-500 bg-gray-50 p-3 rounded">
                        <p className="font-semibold mb-1">💡 Lưu ý:</p>
                        <p>App hiện đang dùng dữ liệu demo ngẫu nhiên. Sau khi tích hợp model thực, kết quả sẽ chính xác dựa trên Deep Learning.</p>
                    </div>
                </div>
            </div>

            <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
        </div>
    );
};

export default EmotionDetector;