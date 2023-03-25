import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # init return dict
    next_probability = {}
    keys = corpus.keys()

    # With probability `1 - damping_factor`, choose link at random chosen from all pages
    # in the corpus with equal probability.
    for key in keys:
        next_probability[key] = (1-damping_factor) / len(keys)

    # get pages linked to current page
    linked_pages = corpus[page]
    if linked_pages:
        for page in linked_pages:
            next_probability[page] += damping_factor / len(linked_pages)

    return next_probability

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # init return dict
    all_pages = list(corpus.keys())
    sample_rank_dict = {}
    for page in all_pages:
        sample_rank_dict[page] = 0 

    # sample step
    step = 1/SAMPLES

    # starting with a page at random
    next_page = random.choice(all_pages) 
    sample_rank_dict[next_page] += step

    # sampling `n` pages according to transition model
    for i in range(n-1):

        # find probability distribution via transition model 
        next_distribution = transition_model(corpus, next_page, damping_factor)
        keys = list(next_distribution.keys())
        values = list(next_distribution.values())

        # choose next sample based on probability distribution
        generator = random.choices(keys, weights=values, k=1)

        # update next page
        next_page = generator[0]
        
        # update sample rank dict
        sample_rank_dict[next_page] += step

    return sample_rank_dict

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # init return dict
    iterate_dict = {}

    # begin by assigning each page a rank of 1 / N
    all_pages = list(corpus.keys())
    total_num_pages = len(all_pages)
    for page in all_pages:
        iterate_dict[page] = 1/total_num_pages

    # repeatedly calculate new rank until convergence
    flag = True
    while flag:
        
        # duplicate new value dict
        new_iterate_dict = copy.deepcopy(iterate_dict)
        
        # calculate new rank values
        for page in all_pages:
            # With probability 1 - d, chose at random and ended up on page p 
            new_iterate_dict[page] = (1-damping_factor)/total_num_pages
            # With probability d, followed a link from a page i to page p.
            linked_probability = 0 
            # Find pages link to current page
            
            # links_to_current = []
            # for is_linked_page in all_pages:
            #     if page in corpus[is_linked_page]:
            #         links_to_current.append(is_linked_page)
            
            links_to_current = [ is_linked_page for is_linked_page in all_pages if page in corpus[is_linked_page] ]
            # exist page link to current page
            if links_to_current:
                for link in links_to_current:
                    linked_probability += iterate_dict[link]/len(corpus[link])
            # no links at all = consider as having one link for every page
            # (including itself).
            # else:
            #     linked_probability = 1/len(all_pages)
            new_iterate_dict[page] += damping_factor*linked_probability

        # Stop condition: all changes <= 0.001
        for page in all_pages:
            if abs(new_iterate_dict[page]-iterate_dict[page]) > 0.001:
                flag =True
                break
            else:
                flag = False
            

        iterate_dict = copy.deepcopy(new_iterate_dict)

    return iterate_dict

            
if __name__ == "__main__":
    main()
