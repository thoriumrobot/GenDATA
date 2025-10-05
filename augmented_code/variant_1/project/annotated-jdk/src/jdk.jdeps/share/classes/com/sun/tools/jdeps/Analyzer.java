/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2013, 2014, Oracle and/or its affiliates. All rights reserved.
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
import com.sun.tools.classfile.Dependency.Location;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;

    @Positive
public class Analyzer {

    @Positive
    public enum Type {

    @Positive
        SUMMARY, MODULE, PACKAGE, CLASS, VERBOSE
    @Positive
    }

    @Positive
    interface Filter {

    @Positive
        boolean accepts(Location origin, Archive originArchive, Location target, Archive targetArchive);
    @Positive
    }

    @Positive
    protected final JdepsConfiguration configuration;

    @Positive
    protected final Type type;

    @Positive
    protected final Filter filter;

    @Positive
    protected final Map<Archive, Dependences> results;

    @Positive
    protected final Map<Location, Archive> locationToArchive;

    @Positive
    boolean run(Iterable<? extends Archive> archives, Map<Location, Archive> locationMap);

    @Positive
    Set<Archive> archives();

    @Positive
    boolean hasDependences(Archive archive);

    @Positive
    Set<String> dependences(Archive source);

    @Positive
    Stream<Archive> requires(Archive source);

    @Positive
    interface Visitor {

    @Positive
        public void visitDependence(String origin, Archive originArchive, String target, Archive targetArchive);
    @Positive
    }

    @Positive
    void visitDependences(Archive source, Visitor v, Type level, Predicate<Archive> targetFilter);

    @Positive
    void visitDependences(Archive source, Visitor v);

    @Positive
    void visitDependences(Archive source, Visitor v, Type level);

    @Positive
    class Dependences implements Archive.Visitor {

    @Positive
        protected final Archive archive;

    @Positive
        protected final Set<Archive> requires;

    @Positive
        protected final Set<Dep> deps;

    @Positive
        protected final Type level;

    @Positive
        protected final Predicate<Archive> targetFilter;

    @Positive
        Set<Dep> dependencies();

    @Positive
        Set<Archive> requires();

    @Positive
        Profile getTargetProfile(Archive target);

    @Positive
        Archive findArchive(Location t);

    @Positive
        @Override
    @Positive
        public void visit(Location o, Location t);

    @Positive
        protected Dep addDep(Location o, Location t);
    @Positive
    }

    @Positive
    class Dep {

    @Positive
        String origin();

    @Positive
        Archive originArchive();

    @Positive
        String target();

    @Positive
        Archive targetArchive();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static boolean notFound(Archive archive);

    @Positive
    static class Jdk8Internals extends Module {

    @Positive
        @Override
    @Positive
        public String name();

    @Positive
        @Pure
    @Positive
        public boolean contains(Location location);

    @Positive
        @Override
    @Positive
        public boolean isJDK();

    @Positive
        @Override
    @Positive
        public boolean isExported(String pn);
    @Positive
    }
    @Positive
}
