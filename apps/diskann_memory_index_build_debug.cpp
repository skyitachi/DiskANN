//
// Created by skyitachi on 25-3-2.
//
#include <omp.h>
#include <cstring>
#include <filesystem>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

int main() {
    uint32_t L = 100;
    uint32_t R = 50;
    uint32_t Lf = 0;
    float alpha = 2;
    uint32_t num_threads = 1;
    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
        .with_filter_list_size(Lf)
        .with_alpha(alpha)
        .with_saturate_graph(false)
        .with_num_threads(num_threads)
        .build();

    auto metric = diskann::Metric::L2;
    size_t data_dim = 128;
    size_t data_num = 10000;
    std::string data_type = "float";
    std::string label_type  = "uint32";
    auto use_opq = false;
    auto use_pq_build = false;
    auto build_PQ_bytes = 0;

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(data_dim)
                      .with_max_points(data_num)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(data_type)
                      .with_label_type(label_type)
                      .is_dynamic_index(false)
                      .with_index_write_params(index_build_params)
                      .is_enable_tags(false)
                      .is_use_opq(use_opq)
                      .is_pq_dist_build(use_pq_build)
                      .with_num_pq_chunks(build_PQ_bytes)
                      .build();

    auto ifp = diskann::IndexFilterParamsBuilder().build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();

    std::string data_path = "test_data/random_128d_10k_vectors.bin";
    std::string index_path = "test_data/random_128d_10k_vectors.index";
    if (std::filesystem::exists(index_path)) {
        index->load(index_path.c_str(), num_threads, L);
    } else {
        index->build(data_path, data_num, ifp);
        index->save(index_path.c_str(), false);
    }

    std::ifstream ifs(data_path, std::ios::binary);
    size_t offset = 8;
    ifs.seekg(offset, std::ios::beg);
    float query[128];
    ifs.read(reinterpret_cast<char*>(query), sizeof(float) * 128);
    int k = 5;
    uint64_t ids[k];
    float distances[k];
    auto [r1, r2] = index->search(query, k, L, ids, distances);
    for (int i = 0; i < k; i++) {
        std::cout <<"closest point: " << ids[i] << ", distance: "<< distances[i] << std::endl;
    }
    std::cout << "r1: " << r1 << ", r2: " << r2 << std::endl;
    return 0;
}

