

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

// clock in navbar

// Calling showTime function at every second
setInterval(showTime, 1000);

// Defining showTime funcion
function showTime() {
	// Getting current time and date
	let time = new Date();
	let hour = time.getHours();
	let min = time.getMinutes();
	let sec = time.getSeconds();
	am_pm = "AM";

	// Setting time for 12 Hrs format
	if (hour >= 12) {
		if (hour > 12) hour -= 12;
		am_pm = "PM";
	} else if (hour == 0) {
		hr = 12;
		am_pm = "AM";
	}
	hour =
		hour < 10 ? "0" + hour : hour;
	min = min < 10 ? "0" + min : min;
	sec = sec < 10 ? "0" + sec : sec;
	let currentTime =
		hour +
		":" +
		min +
		":" +
		sec +
		am_pm;

	// Displaying the time
	document.getElementById(
		"clock"
	).innerHTML = currentTime;
}

showTime();
