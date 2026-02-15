import type { Metadata } from 'next';
import { Space_Grotesk } from 'next/font/google';
import { Header } from '@/components/layout';
import { Providers } from './providers';
import './globals.css';

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk'
});

export const metadata: Metadata = {
  title: 'AI Geneated Backtest',
  description: 'Next.js frontend for the backtest simulation workflow.'
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${spaceGrotesk.variable} font-sans antialiased`}>
        <Providers>
          <Header />
          {children}
        </Providers>
      </body>
    </html>
  );
}
