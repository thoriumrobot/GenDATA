/*
 * Copyright (c) 2017, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jdk.internal.module;

import org.checkerframework.dataflow.qual.Pure;
import java.io.PrintStream;
import java.lang.module.Configuration;
import java.lang.module.ModuleReference;
import java.lang.module.ResolvedModule;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Stream;
import static java.util.stream.Collectors.*;

public class ModuleHashesBuilder {

    public ModuleHashesBuilder(Configuration config, Set<String> modules) {
    }

    public Map<String, ModuleHashes> computeHashes(Set<String> roots);

    static class Graph<T> {

        public Graph(Set<T> nodes, Map<T, Set<T>> edges) {
        }

        public Set<T> nodes();

        public Map<T, Set<T>> edges();

        public Set<T> adjacentNodes(T u);

        @Pure
        public boolean contains(T u);

        public Stream<T> orderedNodes();

        public void ordered(Consumer<T> action);

        public void reverse(Consumer<T> action);

        public Graph<T> transpose();

        public Set<T> dfs(T root);

        public Set<T> dfs(Set<T> roots);

        public void printGraph(PrintStream out);

        static class Builder<T> {

            public void addNode(T node);

            public void addEdge(T u, T v);

            public Graph<T> build();
        }
    }

    private static class TopoSorter<T> {

        public void ordered(Consumer<T> action);

        public void reverse(Consumer<T> action);
    }
}
