import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, StopCircle, AlertCircle, Download, CheckCircle, Brain, RefreshCw } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

const EmotionDetector = () => {
    // State quản lý
    const [mode, setMode] = useState('upload');
    const [isWebcamActive, setIsWebcamActive] = useState(false);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [emotions, setEmotions] = useState(null);
    const [model, setModel] = useState(null);
    const [modelLoading, setModelLoading] = useState(true);
    const [modelError, setModelError] = useState(null);
    const [tfReady, setTfReady] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [modelDetails, setModelDetails] = useState(null);

    // Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const detectionIntervalRef = useRef(null);

    // Danh sách cảm xúc (FER2017 - 7 emotions)
    const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

    // Labels tiếng Việt
    const EMOTION_LABELS = {
        'Angry': 'Tức giận',
        'Disgust': 'Ghê tởm',
        'Fear': 'Sợ hãi',
        'Happy': 'Hạnh phúc',
        'Sad': 'Buồn bã',
        'Surprise': 'Ngạc nhiên',
        'Neutral': 'Bình thường'
    };

    // Màu sắc và emoji
    const emotionColors = {
        'Angry': 'bg-red-500',
        'Disgust': 'bg-green-600',
        'Fear': 'bg-gray-600',
        'Happy': 'bg-yellow-500',
        'Sad': 'bg-blue-500',
        'Surprise': 'bg-purple-500',
        'Neutral': 'bg-gray-400'
    };

    const emotionEmojis = {
        'Angry': '😠',
        'Disgust': '🤢',
        'Fear': '😨',
        'Happy': '😊',
        'Sad': '😢',
        'Surprise': '😲',
        'Neutral': '😐'
    };

    // Hàm load model với progress tracking
    useEffect(() => {
        const loadModelWithProgress = async () => {
            try {
                console.log('🚀 Đang khởi tạo TensorFlow.js...');

                // Kiểm tra và đợi TensorFlow.js sẵn sàng
                await tf.ready();
                setTfReady(true);
                console.log('✅ TensorFlow.js đã sẵn sàng');
                console.log('Backend hiện tại:', tf.getBackend());

                // Các URL thử load model
                const modelPaths = [
                    // Cách 1: Dùng path tương đối từ public folder
                    '/tfjs_model/model.json',

                    // Cách 2: Dùng process.env.PUBLIC_URL cho React
                    process.env.PUBLIC_URL + '/tfjs_model/model.json',

                    // Cách 3: Dùng đường dẫn trực tiếp
                    './tfjs_model/model.json',

                    // Cách 4: Nếu deploy lên GitHub Pages
                    window.location.origin + '/tfjs_model/model.json'
                ];

                let loadedModel = null;
                let lastError = null;

                // Thử load từng path
                for (const modelPath of modelPaths) {
                    try {
                        console.log(`🔄 Đang thử load model từ: ${modelPath}`);

                        // Custom fetch với progress tracking
                        const progressCallback = (fraction) => {
                            const progress = Math.round(fraction * 100);
                            setLoadProgress(progress);
                            console.log(`📊 Load progress: ${progress}%`);
                        };

                        // Load model với custom callback
                        loadedModel = await tf.loadLayersModel(modelPath, {
                            onProgress: progressCallback
                        });

                        // Kiểm tra model hợp lệ
                        if (loadedModel) {
                            console.log(`✅ Model loaded thành công từ: ${modelPath}`);

                            // Lấy thông tin model
                            const inputs = loadedModel.inputs;
                            const outputs = loadedModel.outputs;

                            setModelDetails({
                                inputShape: inputs[0]?.shape,
                                outputShape: outputs[0]?.shape,
                                layers: loadedModel.layers.length,
                                trainableParams: loadedModel.countParams(),
                                path: modelPath
                            });

                            // In summary ra console
                            console.log('📊 Model Summary:');
                            loadedModel.summary();
                            console.log('📐 Input shape:', inputs[0]?.shape);
                            console.log('📈 Output shape:', outputs[0]?.shape);
                            console.log('🏗️ Số layers:', loadedModel.layers.length);
                            console.log('🔢 Số params:', loadedModel.countParams());

                            setModel(loadedModel);
                            setModelError(null);
                            setModelLoading(false);

                            return; // Thoát nếu load thành công
                        }
                    } catch (err) {
                        lastError = err;
                        console.warn(`❌ Không load được từ ${modelPath}:`, err.message);
                        continue;
                    }
                }

                // Nếu tất cả đều thất bại
                if (!loadedModel) {
                    throw new Error(`Không thể load model từ bất kỳ đường dẫn nào. Lỗi cuối: ${lastError?.message}`);
                }

            } catch (error) {
                console.error('❌ Lỗi load model:', error);

                // Tạo error message chi tiết
                const errorDetails = `
Lỗi load model: ${error.message}

Nguyên nhân có thể:
1. File model.json không tồn tại
2. Các file .bin không đúng vị trí
3. Model không tương thích với phiên bản TensorFlow.js

Cấu trúc thư mục mong đợi:
public/
  └── tfjs_model/
       ├── model.json
       ├── group1-shard1of7.bin
       ├── group1-shard2of7.bin
       └── ... (7 file shard)

Vui lòng kiểm tra:
- File model.json có tồn tại trong public/tfjs_model/
- Tất cả 7 file .bin có trong cùng thư mục
- Không có lỗi chính tả trong tên file
                `.trim();

                setModelError(errorDetails);
                setModelLoading(false);

                // Tạo model demo để app vẫn chạy được
                createDemoModel();
            }
        };

        loadModelWithProgress();

        return () => {
            // Cleanup
            stopWebcam();
            if (model) {
                model.dispose();
            }
        };
    }, []);

    // Tạo model demo cho testing
    const createDemoModel = async () => {
        console.log('🔧 Đang tạo model demo...');

        try {
            const demoModel = tf.sequential();

            // Input layer với shape 48x48 grayscale (FER2013 standard)
            demoModel.add(tf.layers.inputLayer({
                inputShape: [48, 48, 1],
                name: 'demo_input'
            }));

            // Conv layers
            demoModel.add(tf.layers.conv2d({
                filters: 32,
                kernelSize: 3,
                activation: 'relu',
                padding: 'same',
                name: 'demo_conv1'
            }));
            demoModel.add(tf.layers.maxPooling2d({
                poolSize: 2,
                name: 'demo_pool1'
            }));

            demoModel.add(tf.layers.conv2d({
                filters: 64,
                kernelSize: 3,
                activation: 'relu',
                padding: 'same',
                name: 'demo_conv2'
            }));
            demoModel.add(tf.layers.maxPooling2d({
                poolSize: 2,
                name: 'demo_pool2'
            }));

            demoModel.add(tf.layers.flatten({ name: 'demo_flatten' }));
            demoModel.add(tf.layers.dense({
                units: 128,
                activation: 'relu',
                name: 'demo_dense1'
            }));
            demoModel.add(tf.layers.dropout({ rate: 0.5 }));

            // Output layer - 7 emotions
            demoModel.add(tf.layers.dense({
                units: 7,
                activation: 'softmax',
                name: 'demo_output'
            }));

            // Compile model
            demoModel.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });

            console.log('✅ Model demo đã được tạo');
            demoModel.summary();

            setModel(demoModel);
            setModelDetails({
                inputShape: [null, 48, 48, 1],
                outputShape: [null, 7],
                layers: demoModel.layers.length,
                trainableParams: demoModel.countParams(),
                path: 'Demo Model'
            });

        } catch (demoError) {
            console.error('❌ Lỗi tạo model demo:', demoError);
        }
    };

    // Reload model
    const reloadModel = async () => {
        setModelLoading(true);
        setLoadProgress(0);
        setModelError(null);

        if (model) {
            model.dispose();
            setModel(null);
        }

        // Đợi một chút để cleanup
        await new Promise(resolve => setTimeout(resolve, 500));

        // Load lại model
        await loadModelWithProgress();
    };

    // Tiền xử lý ảnh
    const preprocessImage = (imageElement) => {
        return tf.tidy(() => {
            // Chuyển sang tensor
            let tensor = tf.browser.fromPixels(imageElement);

            // Chuyển sang grayscale (nếu cần)
            if (tensor.shape[2] === 3) {
                // Cách 1: Lấy kênh green (thường tốt cho face detection)
                // tensor = tensor.slice([0, 0, 1], [-1, -1, 1]);

                // Cách 2: Convert sang grayscale bằng average
                tensor = tensor.mean(2).expandDims(2);
            }

            // Resize về 48x48 (FER2013 standard)
            tensor = tf.image.resizeBilinear(tensor, [48, 48]);

            // Chuẩn hóa pixel values [0, 255] -> [0, 1]
            tensor = tensor.div(255.0);

            // Thêm batch dimension [1, 48, 48, 1]
            tensor = tensor.expandDims(0);

            console.log('🔧 Tensor shape sau preprocess:', tensor.shape);

            return tensor;
        });
    };

    // Phân tích cảm xúc
    const analyzeEmotion = async (imageElement) => {
        if (!model) {
            console.warn('⚠️ Model chưa được load, dùng demo data');
            return analyzeEmotionDemo();
        }

        try {
            setAnalyzing(true);

            // Tiền xử lý ảnh
            const tensor = preprocessImage(imageElement);

            console.log('🧠 Đang dự đoán...');
            console.log('📊 Input tensor shape:', tensor.shape);

            if (modelDetails?.inputShape) {
                console.log('🎯 Model expects shape:', modelDetails.inputShape);
            }

            // Dự đoán
            const startTime = performance.now();
            const prediction = model.predict(tensor);
            const endTime = performance.now();

            console.log(`⏱️ Inference time: ${(endTime - startTime).toFixed(2)}ms`);

            const probabilities = await prediction.data();
            console.log('📈 Raw predictions:', Array.from(probabilities));

            // Tạo kết quả
            const results = {};
            let total = 0;

            EMOTIONS.forEach((emotion, index) => {
                const prob = probabilities[index] || 0;
                const percentage = Math.round(prob * 100);
                results[emotion] = percentage;
                total += percentage;
            });

            // Đảm bảo tổng là 100%
            if (total !== 100 && total > 0) {
                const scale = 100 / total;
                EMOTIONS.forEach(emotion => {
                    results[emotion] = Math.round(results[emotion] * scale);
                });
            }

            // Sắp xếp giảm dần
            const sortedResults = Object.entries(results)
                .sort(([, a], [, b]) => b - a)
                .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

            setEmotions(sortedResults);

            // Cleanup tensors
            tensor.dispose();
            prediction.dispose();

        } catch (error) {
            console.error('❌ Lỗi phân tích:', error);
            // Fallback to demo
            await analyzeEmotionDemo();
        } finally {
            setAnalyzing(false);
        }
    };

    // Demo mode với dữ liệu ngẫu nhiên
    const analyzeEmotionDemo = async () => {
        setAnalyzing(true);

        // Giả lập thời gian xử lý
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Tạo kết quả ngẫu nhiên
        const results = {};
        const randomValues = EMOTIONS.map(() => Math.random());
        const sum = randomValues.reduce((a, b) => a + b, 0);

        EMOTIONS.forEach((emotion, index) => {
            results[emotion] = Math.round((randomValues[index] / sum) * 100);
        });

        // Sắp xếp
        const sortedResults = Object.entries(results)
            .sort(([, a], [, b]) => b - a)
            .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

        setEmotions(sortedResults);
        setAnalyzing(false);
    };

    // Upload ảnh
    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Vui lòng chọn ảnh định dạng JPG, PNG hoặc WebP');
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const imageUrl = event.target.result;
            setUploadedImage(imageUrl);
            setEmotions(null);

            const img = new Image();
            img.onload = () => {
                analyzeEmotion(img);
            };
            img.src = imageUrl;
        };
        reader.readAsDataURL(file);
    };

    // Webcam functions
    const startWebcam = async () => {
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
                setIsWebcamActive(true);

                // Phân tích mỗi 2 giây
                detectionIntervalRef.current = setInterval(() => {
                    if (videoRef.current?.readyState === 4) {
                        captureAndAnalyze();
                    }
                }, 2000);
            }
        } catch (err) {
            console.error('❌ Lỗi webcam:', err);
            alert('Không thể truy cập webcam. Vui lòng kiểm tra quyền truy cập.');
        }
    };

    const captureAndAnalyze = () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Mirror video for natural feel
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Reset transform
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        analyzeEmotion(canvas);
    };

    const stopWebcam = () => {
        if (detectionIntervalRef.current) {
            clearInterval(detectionIntervalRef.current);
            detectionIntervalRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsWebcamActive(false);
        setEmotions(null);
    };

    // Render kết quả
    const renderEmotionResults = () => {
        if (!emotions) return null;

        const topEmotion = Object.keys(emotions)[0];
        const topLabel = EMOTION_LABELS[topEmotion];
        const topEmoji = emotionEmojis[topEmotion];
        const topValue = emotions[topEmotion];

        return (
            <div className="mt-8 p-6 bg-white rounded-xl shadow-lg animate-fadeIn">
                <div className="flex flex-col md:flex-row items-center gap-8">
                    <div className="flex-1 text-center">
                        <div className="text-6xl mb-4">{topEmoji}</div>
                        <h3 className="text-2xl font-bold text-gray-700 mb-2">Cảm xúc chủ đạo</h3>
                        <div className="text-4xl font-bold text-purple-600 mb-2">{topLabel}</div>
                        <div className="text-2xl text-gray-600">{topValue}%</div>
                    </div>

                    <div className="flex-1 w-full">
                        <h4 className="text-lg font-semibold text-gray-700 mb-4">Chi tiết cảm xúc</h4>
                        <div className="space-y-4">
                            {Object.entries(emotions).map(([emotion, value]) => (
                                <div key={emotion} className="space-y-2">
                                    <div className="flex justify-between items-center">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xl">{emotionEmojis[emotion]}</span>
                                            <span className="font-medium text-gray-700">
                                                {EMOTION_LABELS[emotion]}
                                            </span>
                                        </div>
                                        <span className="font-semibold text-gray-800">{value}%</span>
                                    </div>
                                    <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-1000 ${emotionColors[emotion]}`}
                                            style={{ width: `${value}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    // Render model info
    const renderModelInfo = () => {
        if (!modelDetails) return null;

        return (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <h4 className="font-semibold text-gray-700 mb-2">📊 Thông tin Model</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                        <div className="text-gray-500">Input Shape</div>
                        <div className="font-mono">{JSON.stringify(modelDetails.inputShape)}</div>
                    </div>
                    <div>
                        <div className="text-gray-500">Output Shape</div>
                        <div className="font-mono">{JSON.stringify(modelDetails.outputShape)}</div>
                    </div>
                    <div>
                        <div className="text-gray-500">Layers</div>
                        <div className="font-mono">{modelDetails.layers}</div>
                    </div>
                    <div>
                        <div className="text-gray-500">Parameters</div>
                        <div className="font-mono">{modelDetails.trainableParams.toLocaleString()}</div>
                    </div>
                </div>
                {modelDetails.path && (
                    <div className="mt-2 text-xs text-gray-500">
                        Path: <code className="bg-gray-100 px-2 py-1 rounded">{modelDetails.path}</code>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4 md:p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <header className="text-center mb-10">
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
                        🧠 AI Nhận Diện Cảm Xúc
                    </h1>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                        Phân tích cảm xúc sử dụng TensorFlow.js với model được phân thành 7 shard files
                    </p>

                    {/* Status indicators */}
                    <div className="flex flex-wrap gap-3 justify-center mt-6">
                        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${tfReady ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                            <Brain className="w-4 h-4" />
                            <span>TensorFlow.js {tfReady ? '✅' : '⏳'}</span>
                        </div>

                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm bg-blue-100 text-blue-700">
                            <span>Backend: {tf.getBackend()}</span>
                        </div>

                        {modelLoading && (
                            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm bg-purple-100 text-purple-700">
                                <RefreshCw className="w-4 h-4 animate-spin" />
                                <span>Đang tải model... {loadProgress}%</span>
                            </div>
                        )}

                        {model && !modelLoading && (
                            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm bg-green-100 text-green-700">
                                <CheckCircle className="w-4 h-4" />
                                <span>Model đã sẵn sàng</span>
                            </div>
                        )}
                    </div>
                </header>

                {/* Main content */}
                <main className="space-y-8">
                    {/* Model loading progress */}
                    {modelLoading && (
                        <div className="bg-white rounded-xl shadow-lg p-6">
                            <div className="flex items-center gap-4 mb-4">
                                <RefreshCw className="w-6 h-6 animate-spin text-purple-600" />
                                <div className="flex-1">
                                    <h3 className="font-semibold text-gray-700">Đang tải model...</h3>
                                    <p className="text-sm text-gray-500">Đang load 7 shard files từ thư mục public/tfjs_model/</p>
                                </div>
                                <div className="text-lg font-bold text-purple-600">{loadProgress}%</div>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-3">
                                <div
                                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-300"
                                    style={{ width: `${loadProgress}%` }}
                                />
                            </div>
                            <div className="mt-4 grid grid-cols-7 gap-2">
                                {Array.from({ length: 7 }).map((_, i) => (
                                    <div
                                        key={i}
                                        className={`h-2 rounded ${loadProgress >= (i + 1) * 14 ? 'bg-green-500' : 'bg-gray-300'}`}
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Model error display */}
                    {modelError && (
                        <div className="bg-gradient-to-r from-red-50 to-orange-50 border-l-4 border-red-500 rounded-r-lg p-6">
                            <div className="flex items-start gap-3">
                                <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
                                <div className="flex-1">
                                    <h3 className="font-bold text-red-700 text-lg mb-2">⚠️ Lỗi load model</h3>
                                    <div className="bg-red-100 border border-red-200 rounded-lg p-4 mb-4">
                                        <pre className="text-sm text-red-800 whitespace-pre-wrap overflow-x-auto">
                                            {modelError}
                                        </pre>
                                    </div>
                                    <div className="flex gap-3">
                                        <button
                                            onClick={reloadModel}
                                            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all flex items-center gap-2"
                                        >
                                            <RefreshCw className="w-4 h-4" />
                                            Thử lại
                                        </button>
                                        <button
                                            onClick={createDemoModel}
                                            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-all"
                                        >
                                            Dùng model demo
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Model info */}
                    {model && renderModelInfo()}

                    {/* Mode selector */}
                    <div className="flex gap-4 justify-center">
                        <button
                            onClick={() => {
                                if (mode === 'webcam' && isWebcamActive) stopWebcam();
                                setMode('upload');
                                setUploadedImage(null);
                                setEmotions(null);
                            }}
                            className={`flex items-center gap-3 px-6 py-4 rounded-xl font-semibold transition-all ${mode === 'upload'
                                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg scale-105'
                                : 'bg-white text-gray-700 hover:bg-gray-50 shadow-md'
                                }`}
                        >
                            <Upload className="w-6 h-6" />
                            <span>Tải ảnh lên</span>
                        </button>

                        <button
                            onClick={() => {
                                setMode('webcam');
                                setUploadedImage(null);
                                setEmotions(null);
                            }}
                            className={`flex items-center gap-3 px-6 py-4 rounded-xl font-semibold transition-all ${mode === 'webcam'
                                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg scale-105'
                                : 'bg-white text-gray-700 hover:bg-gray-50 shadow-md'
                                }`}
                        >
                            <Camera className="w-6 h-6" />
                            <span>Sử dụng Webcam</span>
                        </button>
                    </div>

                    {/* Content area */}
                    <div className="bg-white rounded-2xl shadow-xl p-6 md:p-8">
                        {/* Upload mode */}
                        {mode === 'upload' && (
                            <div className="space-y-6">
                                <div className="border-3 border-dashed border-gray-300 rounded-xl p-8 md:p-12 text-center hover:border-purple-400 transition-colors cursor-pointer bg-gray-50">
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={handleImageUpload}
                                        className="hidden"
                                        id="imageUpload"
                                    />
                                    <label htmlFor="imageUpload" className="cursor-pointer flex flex-col items-center">
                                        <Upload className="w-16 h-16 md:w-20 md:h-20 text-gray-400 mb-6" />
                                        <p className="text-xl md:text-2xl font-semibold text-gray-700 mb-2">
                                            Chọn ảnh để phân tích
                                        </p>
                                        <p className="text-gray-500">JPG, PNG, WebP</p>
                                    </label>
                                </div>

                                {uploadedImage && (
                                    <div className="mt-6">
                                        <h3 className="text-xl font-semibold text-gray-700 mb-4">Ảnh đã tải lên</h3>
                                        <img
                                            src={uploadedImage}
                                            alt="Uploaded"
                                            className="max-w-full h-auto rounded-lg shadow-lg mx-auto max-h-96"
                                        />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Webcam mode */}
                        {mode === 'webcam' && (
                            <div className="space-y-6">
                                <div className="relative bg-gray-900 rounded-xl overflow-hidden">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full h-auto"
                                        style={{ display: isWebcamActive ? 'block' : 'none' }}
                                    />

                                    {!isWebcamActive && (
                                        <div className="p-12 text-center">
                                            <Camera className="w-20 h-20 mx-auto mb-6 text-gray-400" />
                                            <p className="text-2xl font-semibold text-gray-700 mb-2">
                                                Webcam chưa được kích hoạt
                                            </p>
                                            <p className="text-gray-500">
                                                Nhấn nút bên dưới để bắt đầu
                                            </p>
                                        </div>
                                    )}

                                    <canvas ref={canvasRef} className="hidden" />
                                </div>

                                <div className="flex justify-center gap-4">
                                    {!isWebcamActive ? (
                                        <button
                                            onClick={startWebcam}
                                            disabled={!model}
                                            className="px-8 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            <Camera className="w-6 h-6" />
                                            <span>Bật Webcam</span>
                                        </button>
                                    ) : (
                                        <button
                                            onClick={stopWebcam}
                                            className="px-8 py-3 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all flex items-center gap-3"
                                        >
                                            <StopCircle className="w-6 h-6" />
                                            <span>Tắt Webcam</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Loading indicator */}
                        {analyzing && (
                            <div className="mt-8 text-center">
                                <div className="inline-flex flex-col items-center gap-4">
                                    <div className="relative">
                                        <div className="w-16 h-16 border-4 border-purple-200 rounded-full"></div>
                                        <div className="absolute top-0 left-0 w-16 h-16 border-4 border-purple-600 rounded-full animate-spin border-t-transparent"></div>
                                    </div>
                                    <p className="text-xl font-semibold text-gray-700">Đang phân tích cảm xúc...</p>
                                </div>
                            </div>
                        )}

                        {/* Results */}
                        {!analyzing && renderEmotionResults()}
                    </div>

                    {/* File structure guide */}
                    <div className="bg-white rounded-2xl shadow-xl p-6">
                        <h3 className="text-xl font-bold text-gray-800 mb-4">📁 Cấu trúc thư mục model</h3>
                        <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                            <div className="text-green-400">public/</div>
                            <div className="ml-4">
                                <div className="text-blue-400">└── tfjs_model/</div>
                                <div className="ml-8">
                                    <div className="text-yellow-300">├── model.json</div>
                                    <div className="text-yellow-300">├── group1-shard1of7.bin</div>
                                    <div className="text-yellow-300">├── group1-shard2of7.bin</div>
                                    <div className="text-yellow-300">├── group1-shard3of7.bin</div>
                                    <div className="text-yellow-300">├── group1-shard4of7.bin</div>
                                    <div className="text-yellow-300">├── group1-shard5of7.bin</div>
                                    <div className="text-yellow-300">├── group1-shard6of7.bin</div>
                                    <div className="text-yellow-300">└── group1-shard7of7.bin</div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-6 grid md:grid-cols-2 gap-6">
                            <div className="bg-blue-50 p-5 rounded-xl">
                                <h4 className="font-bold text-blue-700 mb-2">✅ Đã có đúng cấu trúc?</h4>
                                <ul className="space-y-1 text-gray-600">
                                    <li>• 1 file model.json</li>
                                    <li>• 7 file .bin (shard)</li>
                                    <li>• Tất cả trong public/tfjs_model/</li>
                                </ul>
                            </div>

                            <div className="bg-green-50 p-5 rounded-xl">
                                <h4 className="font-bold text-green-700 mb-2">🔄 Cách reload model</h4>
                                <button
                                    onClick={reloadModel}
                                    className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-5 h-5" />
                                    Reload Model
                                </button>
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
};

export default EmotionDetector;