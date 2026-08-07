window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX(button) {
    const entry = button.closest('.bibtex-entry');
    const bibtexElement = entry ? entry.querySelector('pre code') : null;
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Only play the video of the active slide; hidden slides can
                // still intersect the viewport on wide screens.
                const item = video.closest('.item');
                if (item && !item.classList.contains('is-active')) return;
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Fixed-width video carousel: one slide visible, arrows switch slides
function setupResultsCarousel() {
    const carousel = document.querySelector('.results-carousel');
    if (!carousel) return;

    const track = carousel.querySelector('.carousel-track');
    const items = carousel.querySelectorAll('.item');
    const prev = carousel.querySelector('.carousel-arrow-prev');
    const next = carousel.querySelector('.carousel-arrow-next');
    if (!track || items.length === 0 || !prev || !next) return;

    let index = 0;

    // Pagination dots: one per slide, showing position and total
    const dotsContainer = document.createElement('div');
    dotsContainer.className = 'carousel-dots';
    const dots = Array.from(items, function (_, i) {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'carousel-dot';
        dot.setAttribute('aria-label', 'Go to video ' + (i + 1));
        dot.addEventListener('click', function () {
            goTo(i);
        });
        dotsContainer.appendChild(dot);
        return dot;
    });
    carousel.appendChild(dotsContainer);

    function update() {
        // Center the active slide so its neighbors peek at both edges.
        // Percentages resolve against the track (= viewport) width.
        track.style.transform =
            'translateX(calc((100% - var(--carousel-slide-width)) / 2' +
            ' - var(--carousel-gap)' +
            ' - ' + index + ' * (var(--carousel-slide-width) + 2 * var(--carousel-gap))))';
        items.forEach(function (item, i) {
            item.classList.toggle('is-active', i === index);
            const video = item.querySelector('video');
            if (!video) return;
            if (i === index) {
                video.play().catch(function () {});
            } else {
                video.pause();
            }
        });
        dots.forEach(function (dot, i) {
            dot.classList.toggle('is-active', i === index);
        });
    }

    function goTo(i) {
        index = (i + items.length) % items.length;
        update();
    }

    prev.addEventListener('click', function () {
        goTo(index - 1);
    });
    next.addEventListener('click', function () {
        goTo(index + 1);
    });

    // Clicking a peeked neighbor selects it
    items.forEach(function (item, i) {
        item.addEventListener('click', function () {
            if (i !== index) goTo(i);
        });
    });

    update();
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    setupResultsCarousel();

    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})
