var cities = [
    {
        "county": "Baringo",
        "capital": "Kabarnet",
        "code": 30,
        "sub_counties": [
            "Baringo central",
            "Baringo north",
            "Baringo south",
            "Eldama ravine",
            "Mogotio",
            "Tiaty"
        ]
    },
    {
        "county": "Bomet",
        "capital": "Bomet",
        "code": 36,
        "sub_counties": [
            "Bomet central",
            "Bomet east",
            "Chepalungu",
            "Konoin",
            "Sotik"
        ]
    },
    {
        "county": "Bungoma",
        "capital": "Bungoma",
        "code": 39,
        "sub_counties": [
            "Bumula",
            "Kabuchai",
            "Kanduyi",
            "Kimilil",
            "Mt Elgon",
            "Sirisia",
            "Tongaren",
            "Webuye east",
            "Webuye west"
        ]
    },
    {
        "county": "Busia",
        "capital": "Busia",
        "code": 40,
        "sub_counties": [
            "Budalangi",
            "Butula",
            "Funyula",
            "Nambele",
            "Teso North",
            "Teso South"
        ]
    },
    {
        "county": "Elgeyo-Marakwet",
        "capital": "Iten",
        "code": 28,
        "sub_counties": [
            "Keiyo north",
            "Keiyo south",
            "Marakwet east",
            "Marakwet west"
        ]
    },
    {
        "county": "Embu",
        "capital": "Embu",
        "code": 14,
        "sub_counties": [
            "Manyatta",
            "Mbeere north",
            "Mbeere south",
            "Runyenjes"
        ]
    },
    {
        "county": "Garissa",
        "capital": "Garissa",
        "code": 7,
        "sub_counties": [
            "Daadab",
            "Fafi",
            "Garissa",
            "Hulugho",
            "Ijara",
            "Lagdera balambala"
        ]
    },
    {
        "county": "Homa Bay",
        "capital": "Homa Bay",
        "code": 43,
        "sub_counties": [
            "Homabay town",
            "Kabondo",
            "Karachwonyo",
            "Kasipul",
            "Mbita",
            "Ndhiwa",
            "Rangwe",
            "Suba"
        ]
    },
    {
        "county": "Isiolo",
        "capital": "Isiolo",
        "code": 11,
        "sub_counties": [
            "Central",
            "Garba tula",
            "Kina",
            "Merit",
            "Oldonyiro",
            "Sericho"
        ]
    },
    {
        "county": "Kajiado",
        "code": 34,
        "sub_counties": [
            "Isinya.",
            "Kajiado Central.",
            "Kajiado North.",
            "Loitokitok.",
            "Mashuuru."
        ]
    },
    {
        "county": "Kakamega",
        "capital": "Kakamega",
        "code": 37,
        "sub_counties": [
            "Butere",
            "Kakamega central",
            "Kakamega east",
            "Kakamega north",
            "Kakamega south",
            "Khwisero",
            "Lugari",
            "Lukuyani",
            "Lurambi",
            "Matete",
            "Mumias",
            "Mutungu",
            "Navakholo"
        ]
    },
    {
        "county": "Kericho",
        "capital": "Kericho",
        "code": 35,
        "sub_counties": [
            "Ainamoi",
            "Belgut",
            "Bureti",
            "Kipkelion east",
            "Kipkelion west",
            "Soin sigowet"
        ]
    },
    {
        "county": "Kiambu",
        "capital": "Kiambu",
        "code": 22,
        "sub_counties": [
            "Gatundu north",
            "Gatundu south",
            "Githunguri",
            "Juja",
            "Kabete",
            "Kiambaa",
            "Kiambu",
            "Kikuyu",
            "Limuru",
            "Ruiru",
            "Thika town",
            "lari"
        ]
    },
    {
        "county": "Kilifi",
        "capital": "Kilifi",
        "code": 3,
        "sub_counties": [
            "Genzw",
            "Kaloleni",
            "Kilifi north",
            "Kilifi south",
            "Magarini",
            "Malindi",
            "Rabai"
        ]
    },
    {
        "county": "Kirinyaga",
        "capital": "Kerugoya/Kutus",
        "code": 20,
        "sub_counties": [
            "Kirinyaga central",
            "Kirinyaga east",
            "Kirinyaga west",
            "Mwea east",
            "Mwea west"
        ]
    },
    {
        "county": "Kisii",
        "capital": "Kisii",
        "code": 45,
        "sub_counties": [
            
        ]
    },
    {
        "county": "Kisumu",
        "capital": "Kisumu",
        "code": 42,
        "sub_counties": [
            "Kisumu central",
            "Kisumu east ",
            "Kisumu west",
            "Mohoroni",
            "Nyakach",
            "Nyando",
            "Seme"
        ]
    },
    {
        "county": "Kitui",
        "capital": "Kitui",
        "code": 15,
        "sub_counties": [
            "Ikutha",
            "Katulani",
            "Kisasi",
            "Kitui central",
            "Kitui west ",
            "Lower yatta",
            "Matiyani",
            "Migwani",
            "Mutitu",
            "Mutomo",
            "Muumonikyusu",
            "Mwingi central",
            "Mwingi east",
            "Nzambani",
            "Tseikuru"
        ]
    },
    {
        "county": "Kwale",
        "capital": "Kwale",
        "code": 2,
        "sub_counties": [
            "Kinango",
            "Lungalunga",
            "Msambweni",
            "Mutuga"
        ]
    },
    {
        "county": "Laikipia",
        "capital": "Rumuruti",
        "code": 31,
        "sub_counties": [
            "Laikipia central",
            "Laikipia east",
            "Laikipia north",
            "Laikipia west ",
            "Nyahururu"
        ]
    },
    {
        "county": "Lamu",
        "capital": "Lamu",
        "code": 5,
        "sub_counties": [
            "Lamu East",
            "Lamu West"
        ]
    },
    {
        "county": "Machakos",
        "capital": "Machakos",
        "code": 16,
        "sub_counties": [
            "Kathiani",
            "Machakos town",
            "Masinga",
            "Matungulu",
            "Mavoko",
            "Mwala",
            "Yatta"
        ]
    },
    {
        "county": "Makueni",
        "capital": "Wote",
        "code": 17,
        "sub_counties": [
            "Kaiti",
            "Kibwei west",
            "Kibwezi east",
            "Kilome",
            "Makueni",
            "Mbooni"
        ]
    },
    {
        "county": "Mandera",
        "capital": "Mandera",
        "code": 9,
        "sub_counties": [
            "Banissa",
            "Lafey",
            "Mandera East",
            "Mandera North",
            "Mandera South",
            "Mandera West"
        ]
    },
    {
        "county": "Marsabit",
        "capital": "Marsabit",
        "code": 10,
        "sub_counties": [
            "Laisamis",
            "Moyale",
            "North hor",
            "Saku"
        ]
    },
    {
        "county": "Meru",
        "capital": "Meru",
        "code": 12,
        "sub_counties": [
            "Buuri",
            "Igembe central",
            "Igembe north",
            "Igembe south",
            "Imenti central",
            "Imenti north",
            "Imenti south",
            "Tigania east",
            "Tigania west"
        ]
    },
    {
        "county": "Migori",
        "capital": "Migori",
        "code": 44,
        "sub_counties": [
            "Awendo",
            "Kuria east",
            "Kuria west",
            "Mabera",
            "Ntimaru",
            "Rongo",
            "Suna east",
            "Suna west",
            "Uriri"
        ]
    },
    {
        "county": "Mombasa",
        "capital": "Mombasa City",
        "code": 1,
        "sub_counties": [
            "Changamwe",
            "Jomvu",
            "Kisauni",
            "Likoni",
            "Mvita",
            "Nyali"
        ]
    },
    {
        "county": "Murang'a",
        "capital": "Murang'a",
        "code": 21,
        "sub_counties": [
            "Gatanga",
            "Kahuro",
            "Kandara",
            "Kangema",
            "Kigumo",
            "Kiharu",
            "Mathioya",
            "Murang’a south"
        ]
    },
    {
        "county": "Nairobi",
        "capital": "Nairobi City",
        "code": 47,
        "sub_counties": [
            "Dagoretti North Sub County",
            "Dagoretti South Sub County ",
            "Embakasi Central Sub Count",
            "Embakasi East Sub County",
            "Embakasi North Sub County ",
            "Embakasi South Sub County",
            "Embakasi West Sub County",
            "Kamukunji Sub County",
            "Kasarani Sub County ",
            "Kibra Sub County ",
            "Lang'ata Sub County ",
            "Makadara Sub County",
            "Mathare Sub County ",
            "Roysambu Sub County ",
            "Ruaraka Sub County ",
            "Starehe Sub County ",
            "Westlands Sub County "
        ]
    },
    {
        "county": "Nakuru",
        "capital": "Nakuru",
        "code": 32,
        "sub_counties": [
            "Bahati",
            "Gilgil",
            "Kuresoi north",
            "Kuresoi south",
            "Molo",
            "Naivasha",
            "Nakuru town east",
            "Nakuru town west",
            "Njoro",
            "Rongai",
            "Subukia"
        ]
    },
    {
        "county": "Nandi",
        "capital": "Kapsabet",
        "code": 29,
        "sub_counties": [
            "Aldai",
            "Chesumei",
            "Emgwen",
            "Mosop",
            "Namdi hills",
            "Tindiret"
        ]
    },
    {
        "county": "Narok",
        "capital": "Narok",
        "code": 33,
        "sub_counties": [
            "Narok east",
            "Narok north",
            "Narok south",
            "Narok west",
            "Transmara east",
            "Transmara west"
        ]
    },
    {
        "county": "Nyamira",
        "capital": "Nyamira",
        "code": 46,
        "sub_counties": [
            "Borabu",
            "Manga",
            "Masaba north",
            "Nyamira north",
            "Nyamira south"
        ]
    },
    {
        "county": "Nyandarua",
        "capital": "Ol Kalou",
        "code": 18,
        "sub_counties": [
            "Kinangop",
            "Kipipiri",
            "Ndaragwa",
            "Ol Kalou",
            "Ol joro orok"
        ]
    },
    {
        "county": "Nyeri",
        "capital": "Nyeri",
        "code": 19,
        "sub_counties": [
            "Kieni east",
            "Kieni west",
            "Mathira east",
            "Mathira west",
            "Mkurweni",
            "Nyeri town",
            "Othaya",
            "Tetu"
        ]
    },
    {
        "county": "Samburu",
        "capital": "Maralal",
        "code": 25,
        "sub_counties": [
            "Samburu east",
            "Samburu north",
            "Samburu west"
        ]
    },
    {
        "county": "Siaya",
        "capital": "Siaya",
        "code": 41,
        "sub_counties": [
            "Alego usonga",
            "Bondo",
            "Gem",
            "Rarieda",
            "Ugenya",
            "Unguja"
        ]
    },
    {
        "county": "Taita-Taveta",
        "capital": "Voi",
        "code": 6,
        "sub_counties": [
            "Mwatate",
            "Taveta",
            "Voi",
            "Wundanyi"
        ]
    },
    {
        "county": "Tana River",
        "capital": "Hola",
        "code": 4,
        "sub_counties": [
            "Bura",
            "Galole",
            "Garsen"
        ]
    },
    {
        "county": "Tharaka-Nithi",
        "capital": "Chuka",
        "code": 13,
        "sub_counties": [
            "Chuka",
            "Igambangobe",
            "Maara",
            "Muthambi",
            "Tharak north",
            "Tharaka south"
        ]
    },
    {
        "county": "Trans-Nzoia",
        "capital": "Kitale",
        "code": 26,
        "sub_counties": [
            "Cherangany",
            "Endebess",
            "Kiminini",
            "Kwanza",
            "Saboti"
        ]
    },
    {
        "county": "Turkana",
        "capital": "Lodwar",
        "code": 23,
        "sub_counties": [
            "Loima",
            "Turkana central",
            "Turkana east",
            "Turkana north",
            "Turkana south"
        ]
    },
    {
        "county": "Uasin Gishu",
        "capital": "Eldoret",
        "code": 27,
        "sub_counties": [
            "Ainabkoi",
            "Kapseret",
            "Kesses",
            "Moiben",
            "Soy",
            "Turbo"
        ]
    },
    {
        "county": "Vihiga",
        "capital": "Vihiga",
        "code": 38,
        "sub_counties": [
            "Emuhaya",
            "Hamisi",
            "Luanda",
            "Sabatia",
            "vihiga"
        ]
    },
    {
        "county": "Wajir",
        "capital": "Wajir",
        "code": 8,
        "sub_counties": [
            "Eldas",
            "Tarbaj",
            "Wajir East",
            "Wajir North",
            "Wajir South",
            "Wajir West"
        ]
    },
    {
        "county": "West Pokot",
        "capital": "Kapenguria",
        "code": 24,
        "sub_counties": [
            "Central Pokot",
            "North Pokot",
            "Pokot South",
            "West Pokot"
        ]
    }
];

function print_county(county_id){
    var option_str = document.getElementById(county_id);
    option_str.length=0;
    option_str.options[0] = new Option('Select County','');
    option_str.selectedIndex = 0;
    for (var i=0; i<cities.length; i++) {
        option_str.options[option_str.length] = new Option(cities[i].county, i);
    }
}

function print_sub(sub_id, county_index){
    console.log('cities:', cities);
    console.log('county_index:', county_index);
    var option_str = document.getElementById(sub_id);
    option_str.length=0; // Clear the dropdown
    if (county_index > 0) {
        county_index = county_index - 1; // Adjust county_index to account for 'Select County' option
        var sub_arr = cities[county_index].sub_counties;
        console.log('sub_arr:', sub_arr);
        option_str.options[0] = new Option('Select Sub',''); // Add 'Select Sub' option
        for (var i=0; i<sub_arr.length; i++) {
            option_str.options[option_str.length] = new Option(sub_arr[i], sub_arr[i]);
        }
    } else {
        option_str.options[0] = new Option('Select Sub',''); // Add 'Select Sub' option when 'Select County' is selected
    }
}
