/*
 * Copyright (c) 2013, 2014, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdeps;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.tools.classfile.Dependency.Location;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.lang.module.ModuleDescriptor;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Analyzer {

    public enum Type {

        SUMMARY, MODULE, PACKAGE, CLASS, VERBOSE
    }

    interface Filter {

        boolean accepts(Location origin, Archive originArchive, Location target, Archive targetArchive);
    }

    protected final JdepsConfiguration configuration;

    protected final Type type;

    protected final Filter filter;

    protected final Map<Archive, Dependences> results;

    protected final Map<Location, Archive> locationToArchive;

    boolean run(Iterable<? extends Archive> archives, Map<Location, Archive> locationMap);

    Set<Archive> archives();

    boolean hasDependences(Archive archive);

    Set<String> dependences(Archive source);

    Stream<Archive> requires(Archive source);

    interface Visitor {

        public void visitDependence(String origin, Archive originArchive, String target, Archive targetArchive);
    }

    void visitDependences(Archive source, Visitor v, Type level, Predicate<Archive> targetFilter);

    void visitDependences(Archive source, Visitor v);

    void visitDependences(Archive source, Visitor v, Type level);

    class Dependences implements Archive.Visitor {

        protected final Archive archive;

        protected final Set<Archive> requires;

        protected final Set<Dep> deps;

        protected final Type level;

        protected final Predicate<Archive> targetFilter;

        Set<Dep> dependencies();

        Set<Archive> requires();

        Profile getTargetProfile(Archive target);

        Archive findArchive(Location t);

        @Override
        public void visit(Location o, Location t);

        protected Dep addDep(Location o, Location t);
    }

    class Dep {

        String origin();

        Archive originArchive();

        String target();

        Archive targetArchive();

        @Override
        @SuppressWarnings("unchecked")
        public boolean equals(Object o);

        @Override
        public int hashCode();

        public String toString();
    }

    static boolean notFound(Archive archive);

    static class Jdk8Internals extends Module {

        @Override
        public String name();

        @Pure
        public boolean contains(Location location);

        @Override
        public boolean isJDK();

        @Override
        public boolean isExported(String pn);
    }
}
