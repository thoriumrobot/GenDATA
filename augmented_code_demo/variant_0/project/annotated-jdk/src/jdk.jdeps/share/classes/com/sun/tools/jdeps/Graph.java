/*
    @Positive
 * Copyright (c) 2016, 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package com.sun.tools.jdeps;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Deque;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;

    @Positive
public final class Graph<T> {

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
    public Set<Edge<T>> edgesFrom(T u);

    @Positive
    public Graph<T> reduce();

    @Positive
    public Graph<T> reduce(Graph<T> g);

    @Positive
    public Stream<T> orderedNodes();

    @Positive
    public void ordered(Consumer<T> action);

    @Positive
    public void reverse(Consumer<T> action);

    @Positive
    public Graph<T> transpose();

    @Positive
    public Set<T> dfs(Set<T> roots);

    @Positive
    public void printGraph(PrintWriter out);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    static class Edge<T> {

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    static class Builder<T> {

    @Positive
        public void addNode(T node);

    @Positive
        public void addNodes(Set<T> nodes);

    @Positive
        public void addEdge(T u, T v);

    @Positive
        public Graph<T> build();
    @Positive
    }

    @Positive
    static class TopoSorter<T> {

    @Positive
        public void ordered(Consumer<T> action);

    @Positive
        public void reverse(Consumer<T> action);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
