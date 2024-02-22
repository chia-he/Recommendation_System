
/* Background Preloader */
document.addEventListener("DOMContentLoaded", function(){
	$('.preloader-background').delay(50).fadeOut(500);
});

/* Brand Size Control */
$(window).scroll(function() {
  var scrollPosition = $(window).scrollTop();
  if (scrollPosition > 0 ) {
    $(".brand").removeClass("zoomIn");
//    $(".brand").addClass("zoomOut");
    $(".brand").css("transform", "scale(.1)");
  } else {
    $(".brand").addClass("zoomIn");
//    $(".brand").removeClass("zoomOut");
    $(".brand").css("transform", "scale(1)");  
  }
});
