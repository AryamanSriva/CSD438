#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: mpirun -n <n> SK-2210110206.exe <path-to-inp-params.txt>\n";
        MPI_Finalize();
        return 1;
    }

    string inp_path = argv[1];
    int n; double lambda; double alpha; int m;
    vector<vector<int>> adj;
    ifstream in(inp_path);
    if (!in) {
        if (rank==0) cerr << "Cannot open input file: " << inp_path << "\n";
        MPI_Finalize();
        return 1;
    }
    in >> n >> lambda >> alpha >> m;
    adj.assign(n+1, {});
    string line;
    getline(in, line);
    for (int i = 1; i <= n; ++i) {
        if (!getline(in, line)) break;
        if (line.empty()) { --i; continue; }
        istringstream ss(line);
        int v; 
        while (ss >> v) adj[i].push_back(v-1);
    }
    in.close();

    int N = max(n, size);
    mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count() + rank);
    exponential_distribution<double> expd(1.0/lambda);
    uniform_real_distribution<double> uni(0.0, 1.0);
    double p_internal = alpha / (alpha + 1.0);

    vector<int> vc(N,0);
    vector<vector<int>> last_sent(N, vector<int>(N,0));
    int sent_count = 0;
    int internal_count = 0;
    int send_count = 0;

    ostringstream log;
    auto now_str = [&]() {
        auto now = chrono::system_clock::now();
        time_t t = chrono::system_clock::to_time_t(now);
        char buf[64];
        strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&t));
        return string(buf);
    };
    auto now_ms = [&]() {
        using namespace chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    };

    while (sent_count < m) {
        int flag=0; MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        while (flag) {
            if (st.MPI_TAG == 2) {
                int src = st.MPI_SOURCE;
                int seq; MPI_Recv(&seq,1,MPI_INT,src,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                int cnt; MPI_Recv(&cnt,1,MPI_INT,src,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                vector<int> idx(cnt);
                if (cnt>0) MPI_Recv(idx.data(), cnt, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<int> vals(cnt);
                if (cnt>0) MPI_Recv(vals.data(), cnt, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i=0;i<cnt;++i) {
                    int id = idx[i];
                    vc[id] = max(vc[id], vals[i]);
                }
                string mid = string("m") + to_string(src+1) + to_string(seq);
                long long ms = now_ms();
                log << ms << " ";
                log << "Process"<<rank+1<<" receives "<<mid<<" from process"<<src+1<<" at "<<now_str()<<", updates: [";
                for (int i=0;i<cnt;++i) log<<idx[i]+1<<":"<<vals[i]<<" ";
                log<<"]\n";
            } else {
                MPI_Recv(nullptr,0,MPI_CHAR,st.MPI_SOURCE,st.MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        }

        double delay_ms = expd(rng);
        this_thread::sleep_for(chrono::milliseconds((int)round(delay_ms)));

        double r = uni(rng);
            if (r < p_internal) {
            vc[rank]++;
            internal_count++;
            string eid = "e" + to_string(rank+1) + to_string(internal_count);
            long long ms = now_ms();
            log << ms << " ";
            log << "Process"<<rank+1<<" executes internal event "<<eid<<" at "<<now_str()<<", vc: [";
            for (int i=0;i<N;++i) log<<vc[i]<<" ";
            log<<"]\n";
        } else {
            if (adj[rank+1].empty()) {
                vc[rank]++;
                internal_count++;
                string eid = "e" + to_string(rank+1) + to_string(internal_count);
                log << "Process"<<rank+1<<" executes internal event (no neighbors) "<<eid<<" at "<<now_str()<<", vc: [";
                for (int i=0;i<N;++i) log<<vc[i]<<" ";
                log<<"]\n";
            } else {
                vc[rank]++;
                int nei = adj[rank+1][rng()%adj[rank+1].size()];
                vector<int> idx; vector<int> vals;
                double send_prob = 0.7;  // probability to include each changed entry
                for (int i=0;i<N;++i) {
                    if (vc[i] != last_sent[nei][i] && uni(rng) < send_prob) {
                        idx.push_back(i);
                        vals.push_back(vc[i]);
                        last_sent[nei][i] = vc[i];
                    }
                }
                if (idx.empty()) {  // always send at least one entry
                    idx.push_back(rank);
                    vals.push_back(vc[rank]);
                    last_sent[nei][rank] = vc[rank];
                }
                int cnt = (int)idx.size();
                // send seq then cnt then indices and values
                MPI_Send(&send_count,1,MPI_INT,nei,2,MPI_COMM_WORLD);
                MPI_Send(&cnt,1,MPI_INT,nei,2,MPI_COMM_WORLD);
                if (cnt>0) MPI_Send(idx.data(),cnt,MPI_INT,nei,3,MPI_COMM_WORLD);
                if (cnt>0) MPI_Send(vals.data(),cnt,MPI_INT,nei,4,MPI_COMM_WORLD);
                sent_count++;
                send_count++;
                string mid = "m" + to_string(rank+1) + to_string(send_count);
                long long ms = now_ms();
                log << ms << " ";
                log << "Process"<<rank+1<<" sends optimized message "<<mid<<" to process"<<nei+1<<" at "<<now_str()<<", sent_entries="<<cnt<<"\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = chrono::steady_clock::now() + chrono::seconds(2);
    while (chrono::steady_clock::now() < end_time) {
        int flag=0; MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        while (flag) {
            if (st.MPI_TAG == 2) {
                int src = st.MPI_SOURCE;
                int seq; MPI_Recv(&seq,1,MPI_INT,src,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                int cnt; MPI_Recv(&cnt,1,MPI_INT,src,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                vector<int> idx(cnt);
                if (cnt>0) MPI_Recv(idx.data(), cnt, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vector<int> vals(cnt);
                if (cnt>0) MPI_Recv(vals.data(), cnt, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i=0;i<cnt;++i) vc[idx[i]] = max(vc[idx[i]], vals[i]);
                string mid = string("m") + to_string(src+1) + to_string(seq);
                log << "Process"<<rank+1<<" receives "<<mid<<" from process"<<src+1<<" at "<<now_str()<<"\n";
            } else {
                MPI_Recv(nullptr,0,MPI_CHAR,st.MPI_SOURCE,st.MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
        }
        this_thread::sleep_for(chrono::milliseconds(50));
    }

    string s = log.str();
    int len = (int)s.size();
    vector<int> lengths(size);
    MPI_Gather(&len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> displs;
    vector<char> recvbuf;
    if (rank == 0) {
        displs.resize(size);
        int total = 0;
        for (int i=0;i<size;++i) { displs[i]=total; total += lengths[i]; }
        recvbuf.resize(total);
    }
    MPI_Gatherv(s.empty()?nullptr:&s[0], len, MPI_CHAR,
                recvbuf.empty()?nullptr:recvbuf.data(), lengths.data(), displs.empty()?nullptr:displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (rank==0) {
        // combine all entries, parse ms prefix, sort and write ordered log
        vector<pair<long long,string>> entries;
        int offset = 0;
        for (int i=0;i<size;++i) {
            int L = lengths[i];
            string s(recvbuf.data()+offset, recvbuf.data()+offset+L);
            offset += L;
            istringstream iss(s);
            string line;
            while (getline(iss, line)) {
                if (line.empty()) continue;
                size_t pos = line.find(' ');
                if (pos!=string::npos) {
                    string ms_str = line.substr(0,pos);
                    try {
                        long long ms = stoll(ms_str);
                        string rest = line.substr(pos+1);
                        entries.emplace_back(ms, rest);
                    } catch (...) {
                        entries.emplace_back(0, line);
                    }
                } else {
                    entries.emplace_back(0, line);
                }
            }
        }
        sort(entries.begin(), entries.end(), [](auto &a, auto &b){ return a.first < b.first; });
        ofstream out("common_log_SK_2210110206.txt");
        if (!out) cerr<<"Cannot write common_log_SK_2210110206.txt\n";
        else {
            for (auto &p: entries) {
                out << p.second << "\n";
            }
            out.close();
            cout << "Wrote common_log_SK_2210110206.txt\n";
        }
    }

    MPI_Finalize();
    return 0;
}
