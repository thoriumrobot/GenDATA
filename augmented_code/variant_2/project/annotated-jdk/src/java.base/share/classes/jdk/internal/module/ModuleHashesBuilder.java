/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2017, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package jdk.internal.module;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.PrintStream;
    @Positive
import java.lang.module.Configuration;
    @Positive
import java.lang.module.ModuleReference;
    @Positive
import java.lang.module.ResolvedModule;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Deque;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.stream.Stream;
    @Positive
import static java.util.stream.Collectors.*;

    @Positive
public class ModuleHashesBuilder {

    @Positive
    public ModuleHashesBuilder(Configuration config, Set<String> modules) {
    @Positive
    }

    @Positive
    public Map<String, ModuleHashes> computeHashes(Set<String> roots);

    @Positive
    static class Graph<T> {

    @Positive
        public Graph(Set<T> nodes, Map<T, Set<T>> edges) {
    @Positive
        }

    @Positive
        public Set<T> nodes();

    @Positive
        public Map<T, Set<T>> edges();

    @Positive
        public Set<T> adjacentNodes(T u);

    @Positive
        @Pure
    @Positive
        public boolean contains(T u);

    @Positive
        public Stream<T> orderedNodes();

    @Positive
        public void ordered(Consumer<T> action);

    @Positive
        public void reverse(Consumer<T> action);

    @Positive
        public Graph<T> transpose();

    @Positive
        public Set<T> dfs(T root);

    @Positive
        public Set<T> dfs(Set<T> roots);

    @Positive
        public void printGraph(PrintStream out);

    @Positive
        static class Builder<T> {

    @Positive
            public void addNode(T node);

    @Positive
            public void addEdge(T u, T v);

    @Positive
            public Graph<T> build();
    @Positive
        }
    @Positive
    }

    @Positive
    private static class TopoSorter<T> {

    @Positive
        public void ordered(Consumer<T> action);

    @Positive
        public void reverse(Consumer<T> action);
    @Positive
    }
    @Positive
}
