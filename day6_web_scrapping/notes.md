# 🕸️ Web Scraping and Automation – Complete Daily Learning Notes

## 📅 Date: 3rd May 2025
---

## 🧠 Topics Covered
- Introduction to Web Scraping
- Difference between Crawlers and Scrapers
- Setting up Selenium with Firefox (GeckoDriver)
- Automating Browser Tasks with Selenium
- Scraping Dynamic Content using Selenium
- Troubleshooting: NoSuchElement, Timeout, and Access Denied Errors
- HTTP Response Codes Explained (403, 404, etc.)
- Types of Web Scrapers
- Bypassing Access Restrictions

---

## ✅ 1. What is Web Scraping?
Web scraping is the technique used to extract data from websites. It involves making HTTP requests to web pages and parsing the data (HTML, JSON, etc.) to collect specific information.

---

## 🔀 2. Crawler vs Scraper

| Feature           | Crawler                                              | Scraper                                             |
|-------------------|------------------------------------------------------|-----------------------------------------------------|
| Functionality      | Navigates through web pages                         | Extracts specific data from pages                   |
| Example Tool       | Googlebot                                           | Python + Selenium                                   |
| Output             | List of URLs/pages                                  | Structured data (CSV, JSON, Excel, DB)              |
| Scope              | Entire website                                      | Specific elements or sections                       |

---

## 🧰 3. Tools Used

- **Browser**: Firefox
- **Driver**: GeckoDriver (for Firefox)
- **Language**: Python
- **Framework**: Selenium
- **Environment**: Jupyter Notebook

---

## ⚙️ 4. GeckoDriver Installation

1. Download from: [GeckoDriver GitHub](https://github.com/mozilla/geckodriver/releases)
2. Extract and place the `.exe` in a known directory.
3. Add the path (not the `.exe` file) to Environment Variables → System PATH.
4. Test using `geckodriver --version` in CMD.

---

## 💻 5. Selenium Firefox Browser Automation

### Basic Working Script
```python
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
import time

driver = webdriver.Firefox(service=Service())
driver.get("https://www.google.com")
time.sleep(3)
driver.quit()
```
- ✅ Opens Firefox, loads Google, and quits.
- Internally handles browser-driver communication via W3C WebDriver protocol.

---

## 🔍 6. Inspecting and Scraping AmbitionBox

### Attempted Code for Company Names
```python
driver.get("https://www.ambitionbox.com/list-of-companies?page=1")
time.sleep(5)
companies = driver.find_elements(By.CLASS_NAME, "nameDescWrapper")
for company in companies:
    print(company.text)
```
- ❌ `h2.companyName` didn’t work (wrong selector)
- ✅ `div.nameDescWrapper` gave correct results

---

## 🚫 7. Common Errors Faced

### ❌ NoSuchElementException
- The element was not found in DOM.
- Solution: Check for spelling mistakes, wait for page load.

### ❌ TimeoutException
- Element didn't appear within expected wait duration.
- Solution: Increase wait time or check selector.

### ❌ 403 Forbidden
- Server refused access; reasons include:
  - Bot detected
  - Missing headers
  - IP banned

---

## 🌐 8. HTTP Status Codes (Important for Scrapers)

| Code | Name                     | Meaning                                     |
|------|--------------------------|---------------------------------------------|
| 200  | OK                       | Success                                     |
| 403  | Forbidden                | Blocked or not allowed                      |
| 404  | Not Found                | URL incorrect                               |
| 429  | Too Many Requests        | Rate limiting by server                     |
| 500  | Internal Server Error    | Site backend failed                         |
| 503  | Service Unavailable      | Site under maintenance                      |

---

## 📌 9. Bypassing Access Denied (403)

### a. Use Proper Headers
```python
headers = {
  "User-Agent": "Mozilla/5.0 ... Chrome/112.0.0.0 Safari/537.36",
  "Referer": "https://google.com"
}
```

### b. Use Selenium for JS-heavy pages

### c. Add Delays (to behave like human)
```python
import time
time.sleep(3)
```

### d. Use Proxies / VPNs (Advanced)

---

## 🧠 10. Types of Web Scrapers

| Type                         | Description                                  | Tools                        |
|------------------------------|----------------------------------------------|------------------------------|
| Static Page Scraper          | Works with plain HTML                        | Requests, BeautifulSoup      |
| Dynamic Scraper              | Handles JavaScript-generated content         | Selenium, Playwright         |
| API-based Scraper            | Uses public/private APIs                     | Requests, Postman            |
| Headless Browser Scraper     | Runs browser without GUI                     | Selenium (headless mode)     |
| Proxy-based Scraper          | Avoids IP blocks                             | Scrapy + Rotating Proxies    |

---

## 📌 Final Takeaways

- Always inspect elements with **F12 Developer Tools**.
- Use `time.sleep()` or `WebDriverWait` to ensure elements are loaded.
- Use correct `class`, `id`, or other attributes.
- Always follow the website's `robots.txt` and terms of service.

---

## 📁 Summary

| Area              | Status         |
|-------------------|----------------|
| Firefox Launch    | ✅ Success      |
| Element Selection | ✅ Fixed        |
| Blocking Issue    | ❌ 403 Error    |
| Workaround        | 🟡 In Progress  |

---

## 📅 Next Steps

- Explore alternative websites for scraping practice.
- Learn to use Playwright for advanced scraping.
- Study APIs for clean, legal data extraction.
- Try `Scrapy` for large-scale crawling.

---

> Prepared By: Prasant  
> Date: 03-May-2025  
> Goal: Become a master in automation and data scraping 💻