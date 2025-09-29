"""
Optimized NASA Publications PDF Downloader with Manual Download Tracking
Uses Europe PMC directly since it consistently works, with fallbacks and detailed reporting
"""

import os
import re
import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class OptimizedPMCPDFDownloader:
    """Optimized PDF downloader with comprehensive fallback tracking"""
    
    def __init__(self, 
                 output_dir: str = "data/pdfs",
                 delay_between_downloads: float = 1.5,
                 log_file: str = "download_log.json"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.delay = delay_between_downloads
        self.log_file = Path(log_file)
        
        # Initialize download log
        self.download_log = self.load_log()
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def load_log(self) -> Dict[str, Any]:
        """Load existing download log or create new one"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "successful_downloads": [],
            "failed_downloads": [],
            "skipped_downloads": [],
            "download_sessions": []
        }
    
    def save_log(self):
        """Save download log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2, default=str)
    
    def extract_pmc_id(self, pmc_url: str) -> Optional[str]:
        """Extract PMC ID from URL"""
        match = re.search(r'PMC(\d+)', pmc_url)
        return match.group(1) if match else None
    
    def get_all_pdf_urls(self, pmc_id: str) -> List[tuple[str, str]]:
        """Get all possible PDF URLs with source names"""
        urls = [
            (f"https://europepmc.org/articles/pmc{pmc_id}?pdf=render", "Europe PMC"),
            (f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmc_id}&blobtype=pdf", "Europe PMC Backend"),
            (f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/", "PMC Direct"),
            (f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/main.pdf", "PMC Main PDF"),
            (f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/{pmc_id[:2]}/{pmc_id[:4]}/PMC{pmc_id}.pdf", "PMC FTP"),
        ]
        return urls
    
    def test_pdf_url(self, url: str, timeout: int = 10) -> bool:
        """Test if a URL contains a valid PDF"""
        try:
            # Use HEAD request first
            response = self.session.head(url, timeout=timeout, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if any(x in content_type for x in ['pdf', 'octet-stream']):
                    return True
            
            # If HEAD doesn't work, try small GET
            response = self.session.get(url, timeout=timeout, stream=True, headers={'Range': 'bytes=0-1023'})
            if response.status_code in [200, 206]:
                chunk = next(response.iter_content(1024), b'')
                if chunk.startswith(b'%PDF'):
                    return True
                    
        except requests.RequestException:
            pass
        
        return False
    
    def find_working_pdf_url(self, pmc_id: str) -> tuple[Optional[str], Optional[str]]:
        """Find first working PDF URL"""
        urls = self.get_all_pdf_urls(pmc_id)
        
        for url, source in urls:
            if self.test_pdf_url(url):
                return url, source
        
        return None, None
    
    def download_pdf(self, pdf_url: str, filename: str) -> bool:
        """Download a PDF file with validation"""
        file_path = self.output_dir / filename
        
        # Skip if valid file already exists
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    if header.startswith(b'%PDF') and file_path.stat().st_size > 1000:
                        return True
                    else:
                        # Invalid file, remove and re-download
                        file_path.unlink()
            except:
                if file_path.exists():
                    file_path.unlink()
        
        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(x in content_type for x in ['pdf', 'octet-stream', 'binary']):
                return False
            
            # Download to temp file first
            temp_path = file_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate file
            file_size = temp_path.stat().st_size
            if file_size < 1000:
                temp_path.unlink()
                return False
            
            # Check PDF header
            with open(temp_path, 'rb') as f:
                header = f.read(10)
                if not header.startswith(b'%PDF'):
                    temp_path.unlink()
                    return False
            
            # Move to final location
            temp_path.rename(file_path)
            return True
            
        except requests.RequestException:
            temp_path = file_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def download_publications(self, 
                            csv_path: str, 
                            start_from: int = 0) -> Dict[str, Any]:
        """Download PDFs with comprehensive tracking and reporting"""
        
        print(f"🚀 Optimized NASA Publications PDF Downloader")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 Download log: {self.log_file}")
        print("=" * 60)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"📚 Total publications in CSV: {len(df)}")
        
        # Get subset to process
        end_index = len(df)
        df_subset = df.iloc[start_from:end_index]
        
        print(f"📥 Processing {len(df_subset)} PDFs (indices {start_from} to {end_index-1})")
        print()
        
        # Initialize session tracking
        session_info = {
            "timestamp": datetime.now(),
            "start_index": start_from,
            "end_index": end_index-1,
            "requested_count": len(df_subset),
            "successful": [],
            "failed": [],
            "skipped": []
        }
        
        stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'already_exists': 0,
            'invalid_pmc': 0
        }
        
        failed_publications = []  # Track failed ones for manual download
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing"):
            pmc_url = row['Link']
            title = row['Title']
            
            # Extract PMC ID
            pmc_id = self.extract_pmc_id(pmc_url)
            if not pmc_id:
                print(f"\n❌ {idx+1}. Invalid PMC URL: {pmc_url}")
                stats['invalid_pmc'] += 1
                continue
            
            # Create filename
            clean_title = re.sub(r'[<>:"/\\|?*]', '_', title[:50])
            filename = f"PMC{pmc_id}_{clean_title}.pdf"
            
            print(f"\n{idx+1}. PMC{pmc_id}: {title[:60]}...")
            
            stats['attempted'] += 1
            
            # Check if already exists
            file_path = self.output_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        if header.startswith(b'%PDF') and file_path.stat().st_size > 1000:
                            file_size = file_path.stat().st_size
                            print(f"  ⏭️ Already exists ({file_size:,} bytes)")
                            stats['already_exists'] += 1
                            session_info['skipped'].append({
                                "pmc_id": pmc_id,
                                "title": title,
                                "reason": "already_exists"
                            })
                            continue
                except:
                    pass
            
            # Try to find and download PDF
            pdf_url, source = self.find_working_pdf_url(pmc_id)
            
            if not pdf_url:
                print(f"  ❌ No working PDF URL found")
                stats['failed'] += 1
                
                # Add to failed list for manual download
                failed_info = {
                    "index": idx,
                    "pmc_id": pmc_id,
                    "title": title,
                    "pmc_url": pmc_url,
                    "filename": filename,
                    "reason": "no_working_url",
                    "tested_urls": [url for url, source in self.get_all_pdf_urls(pmc_id)]
                }
                failed_publications.append(failed_info)
                session_info['failed'].append(failed_info)
                continue
            
            print(f"  📥 {source}: {pdf_url}")
            
            if self.download_pdf(pdf_url, filename):
                file_size = (self.output_dir / filename).stat().st_size
                print(f"  ✅ Downloaded ({file_size:,} bytes)")
                stats['successful'] += 1
                
                session_info['successful'].append({
                    "pmc_id": pmc_id,
                    "title": title,
                    "filename": filename,
                    "source": source,
                    "file_size": file_size
                })
                
                # Add to global success log
                if pmc_id not in [item['pmc_id'] for item in self.download_log['successful_downloads']]:
                    self.download_log['successful_downloads'].append({
                        "pmc_id": pmc_id,
                        "title": title,
                        "filename": filename,
                        "download_date": datetime.now(),
                        "source": source
                    })
            else:
                print(f"  ❌ Download failed")
                stats['failed'] += 1
                
                failed_info = {
                    "index": idx,
                    "pmc_id": pmc_id,
                    "title": title,
                    "pmc_url": pmc_url,
                    "filename": filename,
                    "reason": "download_failed",
                    "working_url": pdf_url,
                    "source": source
                }
                failed_publications.append(failed_info)
                session_info['failed'].append(failed_info)
            
            # Respectful delay
            time.sleep(self.delay)
        
        # Update logs
        self.download_log['download_sessions'].append(session_info)
        if failed_publications:
            self.download_log['failed_downloads'].extend(failed_publications)
        
        self.save_log()
        
        # Print results
        print(f"\n" + "=" * 60)
        print(f"📊 Download Results:")
        print(f"  Attempted: {stats['attempted']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Already existed: {stats['already_exists']}")
        print(f"  Invalid PMC IDs: {stats['invalid_pmc']}")
        
        if stats['attempted'] > 0:
            success_rate = (stats['successful'] / stats['attempted']) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Report failed downloads for manual handling
        if failed_publications:
            print(f"\n❌ FAILED DOWNLOADS ({len(failed_publications)}) - Manual Download Required:")
            print("-" * 60)
            
            for fail in failed_publications:
                print(f"• PMC{fail['pmc_id']}: {fail['title'][:50]}...")
                print(f"  Original URL: {fail['pmc_url']}")
                print(f"  Reason: {fail['reason']}")
                if fail['reason'] == 'download_failed':
                    print(f"  Working URL: {fail['working_url']}")
                print()
        
        return {**stats, 'failed_publications': failed_publications}
    
    def generate_manual_download_report(self, output_file: str = "manual_download_needed.txt"):
        """Generate a text report of publications that need manual download"""
        
        all_failed = self.download_log.get('failed_downloads', [])
        
        if not all_failed:
            print("🎉 No failed downloads found!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NASA Bioscience Publications - Manual Download Required\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total publications requiring manual download: {len(all_failed)}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for i, fail in enumerate(all_failed, 1):
                f.write(f"{i}. PMC{fail['pmc_id']}\n")
                f.write(f"   Title: {fail['title']}\n")
                f.write(f"   Original PMC URL: {fail['pmc_url']}\n")
                f.write(f"   Reason: {fail['reason']}\n")
                
                if fail['reason'] == 'download_failed' and 'working_url' in fail:
                    f.write(f"   Try this URL: {fail['working_url']}\n")
                elif 'tested_urls' in fail:
                    f.write(f"   URLs tested (all failed):\n")
                    for url in fail['tested_urls']:
                        f.write(f"     - {url}\n")
                
                f.write(f"   Save as: {fail['filename']}\n")
                f.write("\n")
        
        print(f"📄 Manual download report saved to: {output_file}")
    
    def list_downloaded_pdfs(self):
        """List all downloaded PDFs with details"""
        pdf_files = list(self.output_dir.glob("*.pdf"))
        pdf_files.sort()
        
        print(f"📁 Downloaded PDFs ({len(pdf_files)}):")
        total_size = 0
        
        for pdf_file in pdf_files:
            size = pdf_file.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  • {pdf_file.name} ({size_mb:.1f} MB)")
        
        if pdf_files:
            total_mb = total_size / (1024 * 1024)
            print(f"\nTotal: {len(pdf_files)} files, {total_mb:.1f} MB")


def main():
    """Main function to test the downloader"""
    csv_path = "data/SB_publication_PMC.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    downloader = OptimizedPMCPDFDownloader(
        output_dir="data/pdfs",
        delay_between_downloads=1.0,
        log_file="nasa_pdf_download_log.json"
    )
    
    # Download publications
    print("Testing optimized downloader with comprehensive tracking...")
    stats = downloader.download_publications(
        csv_path=csv_path,
        start_from=0
    )
    
    # Generate manual download report
    downloader.generate_manual_download_report("manual_downloads_needed.txt")
    
    # List what we have
    print()
    downloader.list_downloaded_pdfs()
    
    print(f"\n🔍 Check 'manual_downloads_needed.txt' for publications that need manual download")
    print(f"📋 Full log available in 'nasa_pdf_download_log.json'")


if __name__ == "__main__":
    main()
