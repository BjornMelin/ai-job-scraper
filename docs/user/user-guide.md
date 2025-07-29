# 📖 User Guide: AI Job Scraper

This comprehensive guide covers all features and functionality of the AI Job Scraper dashboard, helping you maximize your job search efficiency.

## 🏠 Dashboard Overview

The AI Job Scraper features a clean, intuitive interface built with Streamlit. The main dashboard consists of:

### Main Components

- **Header**: Application title and rescrape button

- **Sidebar**: Global filters and company management

- **Main Area**: Tabbed job views with different display modes

- **Footer**: Quick statistics summary

### Color Scheme & Design

The application uses a professional tech-inspired theme:

- **Dark gradient background** for reduced eye strain

- **Blue accent colors** for interactive elements

- **Responsive design** that works on desktop and mobile

- **Card-based layouts** for easy scanning

## 🔍 Global Filtering System

The sidebar contains powerful filtering options that apply across all tabs:

### Company Filter

```text
Companies: [ ] Anthropic [ ] OpenAI [x] NVIDIA [ ] Meta...
```

- **Multiselect dropdown** with all available companies

- **Select specific companies** to focus your search

- **"All" option** shows jobs from every company

- **Dynamic updates** as you add/remove companies

### Keyword Search

```text
Keyword Search: [AI Engineer Senior Remote    ]
```

- **Full-text search** across job titles and descriptions

- **Case-insensitive** matching

- **Supports partial matches** and multiple terms

- **Real-time filtering** as you type

### Date Range Filtering

```text
Posted From: [📅 2024-01-01] Posted To: [📅 2024-12-31]
```

- **Date picker widgets** for precise range selection

- **Filter by posting date** to find recent opportunities

- **Leave blank** for no date restrictions

- **Useful for tracking** new vs. older postings

## 📋 Job View Modes

### List View (Table Format)

**Best for**: Bulk editing, quick scanning, data export

```text
┌─────────────────────────────────────────────────────────────┐
│ Company │ Title              │ Location │ Favorite │ Status  │
├─────────────────────────────────────────────────────────────┤
│ OpenAI  │ AI Research Eng... │ SF       │ ☐        │ New ▼   │
│ NVIDIA  │ ML Infrastructure  │ Remote   │ ☑        │ Applied │
└─────────────────────────────────────────────────────────────┘
```

**Features:**

- **Editable table** using Streamlit's data_editor

- **Checkbox column** for favorites (⭐)

- **Dropdown column** for status (New/Interested/Applied/Rejected)  

- **Text column** for personal notes

- **Link column** with "Apply" buttons

- **Save button** to persist all changes to database

### Card View (Visual Grid)

**Best for**: Visual browsing, detailed job information, mobile viewing

```text
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ OpenAI: AI Research Engineer│ │ NVIDIA: ML Infrastructure   │
│ Develop next-generation...  │ │ Build scalable ML systems...│
│ Location: SF | Posted: 2d   │ │ Location: Remote | Posted:1w│  
│ Status: New | Favorite:     │ │ Status: Applied | Favorite:⭐│
│ [Apply] [Toggle Fav] [Edit] │ │ [Apply] [Toggle Fav] [Edit] │
└─────────────────────────────┘ └─────────────────────────────┘
```

**Features:**

- **3-column grid** layout (adjusts on mobile)

- **Job description preview** (first 150 characters)

- **Visual status indicators** and favorite stars

- **Interactive widgets** for quick updates:
  - Toggle Favorite button
  - Status dropdown (changes immediately)
  - Notes text area (auto-saves on change)

- **Pagination controls** (9 jobs per page)

- **Sorting options** (by Posted Date, Title, or Company)

## 📑 Tab Organization

### 📋 All Jobs Tab

**Purpose**: Browse all scraped job postings

- **Shows every job** from all companies (filtered by global filters)

- **Default landing page** when opening the application

- **Full dataset access** for comprehensive searching

- **Best for initial exploration** and broad filtering

**Pro tip**: Use this tab with company filters to see everything from specific organizations.

### ⭐ Favorites Tab  

**Purpose**: Track jobs you're most interested in

- **Only shows jobs marked** with the favorite star

- **Persists across sessions** (saved in database)

- **Quick access** to high-priority opportunities  

- **Ideal for shortlisting** positions for applications

**Workflow**: Mark interesting jobs as favorites in "All Jobs", then use this tab for focused review.

### ✅ Applied Tab

**Purpose**: Monitor your application pipeline

- **Shows jobs with status = "Applied"**

- **Track application progress** and follow-ups

- **Export for records** and interview preparation

- **Avoid duplicate applications**

**Integration**: Change job status to "Applied" after submitting applications to track them here.

## ⚙️ Company Management

Located in the sidebar under "Manage Companies":

### Company List

```text
┌─────────────────────────────────────────┐
│ ID │ Name      │ URL               │ Active │
├─────────────────────────────────────────┤
│ 1  │ OpenAI    │ openai.com/car... │ ☑      │
│ 2  │ Anthropic │ careers.anthro... │ ☑      │  
│ 3  │ Custom Co │ company.com/jobs  │ ☐      │
└─────────────────────────────────────────┘
```

### Managing Companies

**Activate/Deactivate Companies:**

- **Check/uncheck** the "Active" checkbox

- **Only active companies** are scraped during rescrape operations

- **Useful for focusing** on specific companies or reducing scrape time

**Add New Companies:**

1. **Enter company name** in "Add New Company Name" field
2. **Enter careers page URL** in "Add New URL" field
3. **Click "Save Companies"** to add to database
4. **New companies default** to active status

**Best Practices:**

- **Use descriptive names** (e.g., "Google DeepMind" not just "Google")

- **Verify URLs** point to actual job listings pages

- **Test with small companies first** before adding many at once

## 🔄 Scraping Operations

### Manual Rescrape

**Location**: Main dashboard, prominent blue button

```text
[🔄 Rescrape Jobs]
```

**What happens when you click:**

1. **Spinner appears** with "Scraping..." message
2. **Active companies** are processed in parallel
3. **Progress shown** in terminal/logs (if visible)
4. **Database updated** with new/changed/removed jobs
5. **Success message** or error notification appears
6. **UI refreshes** to show updated job counts

**Performance Expectations:**

- **First run**: 45-90 seconds (no cache)

- **Subsequent runs**: 15-45 seconds (with cache hits)

- **Cache hit rate**: 70-90% after first run

- **Typical results**: 20-150 jobs depending on market conditions

### Automatic Cache System

The application includes intelligent caching for performance:

**Cache Behavior:**

- **Schema caching** learns site structures automatically

- **File-based storage** in `./cache/` directory  

- **90% speed improvement** on cached sites

- **Graceful fallback** to LLM extraction when cache fails

- **Auto-regeneration** when sites change structure

**Cache Files:**

```text
cache/
├── anthropic.json    # Cached extraction schema
├── openai.json
├── nvidia.json
└── ...
```

## 📊 Data Management

### Job Status Workflow

**Recommended Status Flow:**

1. **New** → Freshly scraped, not yet reviewed
2. **Interested** → Reviewed and potentially worth applying
3. **Applied** → Application submitted
4. **Rejected** → Application declined or position closed

**Status Management:**

- **Change via dropdown** in both list and card views

- **Immediately saved** to database

- **Filter by status** using the Applied tab for applied jobs

- **Export by status** using CSV download

### Notes System

**Adding Notes:**

- **List view**: Edit in the Notes column, click Save Changes

- **Card view**: Use the text area widget (auto-saves)

- **Unlimited length** but keep concise for UI performance

- **Supports markdown** when exported

**Note Ideas:**

- Contact information for referrals

- Application deadlines and requirements

- Interview insights and feedback

- Salary ranges and benefits notes

- Follow-up reminders

### Data Export

**CSV Export** available in each tab:

```text
[📥 Export CSV]
```

**Export Contents:**

- **All visible jobs** in current tab after filtering

- **All columns** including notes, status, favorites

- **Formatted dates** and clean data

- **Opens in Excel/Google Sheets** for further analysis

**Use Cases:**

- **Backup job data** before major changes

- **Share with career counselors** or advisors

- **Import into personal CRM** systems

- **Create application tracking** spreadsheets

## 🔍 Advanced Filtering & Search

### Per-Tab Search

Each tab includes a search box:

```text
Search in this tab: [machine learning remote     ]
```

**Search Behavior:**

- **Scoped to current tab** (All/Favorites/Applied)

- **Searches both title and description** fields

- **Case-insensitive** partial matching

- **Real-time results** as you type

- **Combines with global filters** for refined results

### Search Strategies

**Finding Remote Jobs:**

```text
Keywords: "remote" OR "distributed" OR "anywhere"
```

**Senior Positions:**

```text
Keywords: "senior" OR "staff" OR "principal" OR "lead"
```

**Specific Technologies:**

```text
Keywords: "pytorch" OR "tensorflow" OR "kubernetes"
```

**Location-Based:**

```text
Keywords: "san francisco" OR "seattle" OR "new york"
```

### Combining Filters

#### Example: Remote Senior ML Jobs at FAANG

1. **Company Filter**: Select Meta, Microsoft, etc.
2. **Global Keyword**: "senior remote"
3. **Per-tab Search**: "machine learning"
4. **Date Filter**: Last 30 days

## 📱 Mobile Experience

The application is fully responsive and works well on mobile devices:

### Mobile Optimizations

- **Single column** layout on small screens

- **Touch-friendly** buttons and controls  

- **Readable text** sizes and spacing

- **Simplified navigation** with collapsible sidebar

- **Card view recommended** for mobile browsing

### Mobile Workflow

1. **Use card view** for better visual experience
2. **Global filters** still available via sidebar
3. **Toggle favorite** directly on cards
4. **Status updates** work immediately
5. **Apply buttons** open in new tabs

## 📈 Performance Dashboard

### Statistics Panel (Bottom of Page)

```text
Stats 📈
Total Jobs: 127
Favorites: 15
Applied: 8
```

**Metrics Explained:**

- **Total Jobs**: All jobs currently in database (after global filters)

- **Favorites**: Count of jobs marked with favorite star

- **Applied**: Count of jobs with "Applied" status

### Session Performance

**Check terminal output** for detailed performance metrics:

```text
📊 Session Summary:
  Duration: 32.1s
  Companies: 7  
  Jobs found: 89
  Cache hit rate: 87%
  LLM calls: 1
  Errors: 0
```

**Optimization Indicators:**

- **High cache hit rate** (>70%) = Good performance

- **Low LLM calls** (<3) = Cost efficient  

- **Zero errors** = Stable scraping

- **Short duration** (<60s) = Responsive experience

## 🎯 Workflow Recommendations

### Daily Job Hunter Routine

1. **Morning Check** (5 minutes)
   - Open application
   - Click "Rescrape Jobs"
   - Review "All Jobs" tab for new postings
   - Mark interesting positions as favorites

2. **Weekly Review** (30 minutes)
   - Review "Favorites" tab thoroughly
   - Research company backgrounds
   - Update status for applied positions
   - Export CSV for record keeping

3. **Monthly Maintenance** (15 minutes)
   - Add/remove companies based on interests
   - Clean up notes and outdated favorites
   - Update job search criteria
   - Review performance metrics

### Application Tracking Workflow

**Before Applying:**

1. Mark job as **"Interested"** status
2. Add **application notes** (deadline, requirements)
3. Export **CSV for application materials** preparation

**After Applying:**  

1. Change status to **"Applied"**
2. Add **application date and method** to notes
3. Track in **Applied tab** for follow-ups

**After Response:**

1. Update status to **"Rejected"** if declined
2. Add **interview feedback** to notes
3. Keep for future reference and learning

## 🔧 Customization Tips

### Optimizing for Your Search

**For Specific Roles:**

- Edit company list to focus on relevant organizations

- Use targeted keywords in global search

- Set up date filters for fresh postings only

**For Location Preferences:**

- Add location terms to keyword search

- Focus on companies with known remote policies

- Use notes to track location flexibility

**For Experience Level:**

- Include seniority keywords (junior, mid-level, senior)

- Filter by company types (startup vs. enterprise)

- Track compensation ranges in notes

### Privacy & Data Control

**Local-First Approach:**

- All data stored locally in SQLite database

- No external data sharing except OpenAI API (optional)

- Full control over company selection and data retention

**Data Portability:**

- CSV export for backup and migration

- Standard SQLite format for database access

- All configurations stored in local files

**Security Considerations:**

- Keep OpenAI API keys secure in .env file

- Regular CSV backups recommended

- No personal data sent to scraped companies

Now you're ready to make the most of AI Job Scraper! For technical details and customization options, check out the [Developer Guide](DEVELOPER_GUIDE.md).
