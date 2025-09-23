/* 
 * Fix for collapse scroll positioning issue
 * This script improves the scroll behavior when collapsing code blocks
 * for better reading continuity.
 * 
 * The issue: when clicking collapse, the code block gets positioned too high
 * in the viewport, interrupting the reading flow.
 * 
 * The solution: position the collapsed code block so its bottom is roughly
 * 1/3 from the top of the viewport, leaving visible content below for
 * continuous reading.
 */

document.addEventListener("DOMContentLoaded", function () {
  // Wait for the theme's JavaScript to load and set up collapse functionality
  setTimeout(() => {
    console.log('Applying collapse scroll behavior fix...');
    
    const collapsableCodeToggles = document.querySelectorAll(
      "div.cell[class*='tag_collapse'] .collapse-toggle-bar",
    );
    
    console.log(`Found ${collapsableCodeToggles.length} collapse toggles to fix`);
    
    // Remove existing event listeners by cloning and replacing elements
    for (let i = 0; i < collapsableCodeToggles.length; i++) {
      const oldToggle = collapsableCodeToggles[i];
      const newToggle = oldToggle.cloneNode(true);
      oldToggle.parentNode.replaceChild(newToggle, oldToggle);
      
      // Add improved event listener
      newToggle.addEventListener("click", function (e) {
        console.log('Collapse toggle clicked with improved behavior');
        e.preventDefault();
        var codeBlock = this.closest("div.cell[class*='tag_collapse']");
        var codeBlockH = codeBlock.querySelector(".highlight");
        var indicator = this.querySelector(".collapse-indicator");

        if (codeBlock.classList.contains("expanded")) {
          console.log('Collapsing code block...');
          codeBlock.classList.remove("expanded");
          indicator.textContent = "Expand";
          
          // Apply height based on collapse class
          const collapseAccToHeight = (classList, elH) => {
            for (let className of classList) {
              if (className.startsWith("tag_collapse-")) {
                const index = className.indexOf("-");
                const height = className.substring(index + 1);
                if (height && !isNaN(height)) {
                  elH.style.height = parseInt(height) + 0.5 + "em";
                  return true;
                }
              }
            }
            return false;
          };
          
          collapseAccToHeight(codeBlock.classList, codeBlockH);
          
          // Improved scroll behavior for better reading continuity
          setTimeout(() => {
            console.log('Applying improved scroll positioning...');
            const rect = codeBlock.getBoundingClientRect();
            const viewportHeight = window.innerHeight;
            
            // Position the collapsed code block so the reader can continue reading
            // Position the bottom of the collapsed block about 30% from the top of viewport
            // This leaves about 70% of the viewport showing content below the collapsed block
            const targetPositionFromTop = viewportHeight * 0.3;
            const currentScrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const elementTop = rect.top + currentScrollTop;
            const elementHeight = rect.height;
            
            // Calculate scroll position to put the bottom of collapsed block at target position
            const newScrollTop = elementTop + elementHeight - targetPositionFromTop;
            
            // Ensure we don't scroll past page boundaries
            const maxScrollTop = Math.max(0, document.documentElement.scrollHeight - viewportHeight);
            const finalScrollTop = Math.max(0, Math.min(newScrollTop, maxScrollTop));
            
            console.log(`Scrolling to position ${finalScrollTop} for optimal reading flow`);
            
            window.scrollTo({
              top: finalScrollTop,
              behavior: 'smooth'
            });
          }, 150); // Slightly longer delay for height change to complete
        } else {
          console.log('Expanding code block...');
          codeBlock.classList.add("expanded");
          indicator.textContent = "Collapse";
          codeBlockH.style.height = "auto";
        }
      });
    }
  }, 1500); // Wait for theme JS to fully load
});