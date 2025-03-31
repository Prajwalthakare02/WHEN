// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
    console.log('When Portal loaded successfully!');
    
    // Add a simple animation to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    
    if (featureCards.length > 0) {
        featureCards.forEach((card, index) => {
            setTimeout(() => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 100);
            }, index * 200);
        });
    }
}); 