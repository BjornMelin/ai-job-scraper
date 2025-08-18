# URL-Based State Management Implementation

## Overview

Implemented comprehensive URL-based state management for the AI Job Scraper using Streamlit's `st.query_params` API. This enables shareable job searches, bookmarkable views, and browser history support.

## Features Implemented

### 1. Filter State Persistence (`src/ui/utils/url_state.py`)

**Core Functions:**

- `sync_filters_from_url()` - Reads filter parameters from URL and updates session state

- `update_url_from_filters()` - Updates URL parameters when filters change

- `validate_url_params()` - Validates and sanitizes URL parameters

- `clear_url_params()` - Clears all URL parameters

**Supported Filter Parameters:**

- `keyword` - Search terms

- `company` - Comma-separated company list

- `salary_min` - Minimum salary filter

- `salary_max` - Maximum salary filter  

- `date_from` - Start date filter (ISO format)

- `date_to` - End date filter (ISO format)

### 2. Job Tab Selection Persistence

**Parameter:**

- `tab` - Selected tab ("all", "favorites", "applied")

**Implementation:**

- Button-based tab selection instead of native st.tabs for programmatic control

- Real-time URL sync when tabs are changed

- Deep linking to specific tabs

### 3. Company Selection Persistence

**Parameter:**

- `selected` - Comma-separated company IDs for bulk operations

**Features:**

- Persistent selection across page refreshes

- Bulk selection operations sync with URL

- Bookmarkable company selection states

### 4. URL Parameter Validation

**Validation Rules:**

- Salary values: 0-500,000 range

- Dates: Between 2020 and reasonable future dates

- Tab values: Must be "all", "favorites", or "applied"

- Company IDs: Must be valid integers

**Error Handling:**

- Invalid parameters are ignored with warnings

- Graceful fallback to default values

- User-friendly error messages

## File Changes

### `src/ui/utils/url_state.py` (NEW)

Complete URL state management utility module with sync, validation, and helper functions.

### `src/ui/components/sidebar.py`

- Added URL sync on filter changes

- Integrated `sync_filters_from_url()` on sidebar render

- Clear URL params when filters are cleared

### `src/ui/pages/jobs.py`  

- Added tab selection persistence with URL sync

- Replaced native st.tabs with custom button-based tabs

- Added URL validation with user warnings

- Added "Share Filters" button for easy URL sharing

### `src/ui/pages/companies.py`

- Added company selection persistence

- URL sync on selection changes

- Bulk operation URL sync

- URL validation integration

## URL Parameter Schema

### Job Filters

```
?keyword=python&company=openai,anthropic&salary_min=100000&salary_max=200000&date_from=2024-01-01&date_to=2024-12-31&tab=favorites
```

### Company Management

```
?selected=1,2,3,4
```

## Usage Examples

### Shareable Job Searches

- Apply filters in sidebar

- Select tab preference  

- Click "Share Filters" to get shareable URL

- Share URL preserves all filter state and tab selection

### Bookmarkable Views

- Set filters and tab selection

- Bookmark the URL

- Returning to bookmark restores exact view state

### Browser Navigation

- Back/forward buttons work with filter changes

- Each filter change creates new browser history entry

- Tab selection preserved in browser history

## Benefits

1. **Enhanced User Experience**
   - Bookmarkable searches and views
   - Shareable filter combinations
   - Browser back/forward navigation support

2. **Improved Collaboration**
   - Share specific job searches with team members
   - Bookmark commonly used filter combinations
   - Persistent state across browser sessions

3. **Better Analytics Potential**
   - Track popular filter combinations via URL parameters
   - Understand user search patterns
   - Deep link support for external integrations

## Technical Implementation Notes

- Uses Streamlit's native `st.query_params` API for URL management

- All URL parameters are optional with sensible defaults

- Parameters only added to URL when different from defaults (clean URLs)

- Comprehensive input validation and sanitization

- Error handling with user-friendly feedback

- Library-first approach following project conventions

## Future Enhancements

Potential improvements that could be added:

1. **Saved Searches** - Allow users to save filter combinations with custom names
2. **Search History** - Track and provide quick access to recent filter combinations
3. **Advanced Sharing** - QR codes or shortened URLs for easier sharing
4. **URL Presets** - Pre-configured filter combinations for common use cases
5. **Analytics Integration** - Track usage patterns via URL parameters
