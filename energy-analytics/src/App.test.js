import { render, screen } from '@testing-library/react';
import LandingSkeleton from './skeleton_pages/LandingSkeleton';

test('renders the landing skeleton copy', () => {
  render(<LandingSkeleton />);
  expect(screen.getByText(/preparing the landing experience/i)).toBeInTheDocument();
});
