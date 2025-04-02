const CACHE_NAME = "skin-cancer-pwa-v1";
const urlsToCache = [
    "/",
    "/static/css/style.css",
    "/static/js/script.js",
    "/static/icons/icon-192x192.png",
    "/static/icons/icon-512x512.png"
];

// Install service worker and cache resources
self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(urlsToCache);
        })
    );
});

// static/service-worker.js
self.addEventListener('install', (event) => {
    console.log('Service Worker installed');
});


// Fetch cached resources when offline
self.addEventListener("fetch", (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});
