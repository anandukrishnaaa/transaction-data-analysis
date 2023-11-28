

// function to switch to light and dark mode
let darkModeIcon = document.querySelector('#darkMode-icon');

darkModeIcon.onclick = () => {
    darkModeIcon.classList.toggle('bxs-sun');
    document.body.classList.toggle('dark-mode')
}

// java script for welcome user animation


$(function () {

    var welcomeSection = $('.welcome-section'),
        enterButton = welcomeSection.find('.enter-button');

    setTimeout(function () {
        welcomeSection.removeClass('content-hidden');
    }, 500);

    // enterButton.on('click', function(e) {
    //     e.preventDefault();
    //     welcomeSection.addClass('content-hidden').fadeOut();
    // });
});


// show and hide the table 1 in train ml model in ml.html

function ShowAndHide() {
    var x = document.getElementById('trainMlTable1');
    if (x.style.display == 'none') {
        x.style.display = 'block';
    } else {
        x.style.display = 'none';
    }
}

