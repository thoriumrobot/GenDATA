/*
    @Positive
 * Copyright (c) 2012, 2014, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.tools.classfile.Dependency.Location;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.net.URI;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.stream.Stream;
    @Positive
import static com.sun.tools.jdeps.Module.trace;

    @Positive
public class Archive implements Closeable {

    @Positive
    public static Archive getInstance(Path p, Runtime.Version version);

    @Positive
    protected Map<Location, Set<Location>> deps;

    @Positive
    protected Archive(String name) {
    @Positive
    }

    @Positive
    protected Archive(String name, URI location, ClassFileReader reader) {
    @Positive
    }

    @Positive
    protected Archive(Path p, ClassFileReader reader) {
    @Positive
    }

    @Positive
    public ClassFileReader reader();

    @Positive
    public String getName();

    @Positive
    public Module getModule();

    @Positive
    @Pure
    @Positive
    public boolean contains(String entry);

    @Positive
    public void addClass(Location origin);

    @Positive
    public void addClass(Location origin, Location target);

    @Positive
    public Set<Location> getClasses();

    @Positive
    public Stream<Location> getDependencies();

    @Positive
    public boolean hasDependences();

    @Positive
    public void visitDependences(Visitor v);

    @Positive
    public boolean isEmpty();

    @Positive
    public String getPathName();

    @Positive
    public Optional<Path> path();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public static boolean isSameLocation(Archive archive, Archive other);

    @Positive
    @Override
    @Positive
    public void close() throws IOException;

    @Positive
    interface Visitor {

    @Positive
        void visit(Location origin, Location target);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
