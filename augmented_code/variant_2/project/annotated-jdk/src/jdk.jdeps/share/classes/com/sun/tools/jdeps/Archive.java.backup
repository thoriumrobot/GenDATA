/*
 * Copyright (c) 2012, 2014, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.tools.classfile.Dependency.Location;
import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import static com.sun.tools.jdeps.Module.trace;

public class Archive implements Closeable {

    public static Archive getInstance(Path p, Runtime.Version version);

    protected Map<Location, Set<Location>> deps;

    protected Archive(String name) {
    }

    protected Archive(String name, URI location, ClassFileReader reader) {
    }

    protected Archive(Path p, ClassFileReader reader) {
    }

    public ClassFileReader reader();

    public String getName();

    public Module getModule();

    @Pure
    public boolean contains(String entry);

    public void addClass(Location origin);

    public void addClass(Location origin, Location target);

    public Set<Location> getClasses();

    public Stream<Location> getDependencies();

    public boolean hasDependences();

    public void visitDependences(Visitor v);

    public boolean isEmpty();

    public String getPathName();

    public Optional<Path> path();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public String toString();

    public static boolean isSameLocation(Archive archive, Archive other);

    @Override
    public void close() throws IOException;

    interface Visitor {

        void visit(Location origin, Location target);
    }
}
