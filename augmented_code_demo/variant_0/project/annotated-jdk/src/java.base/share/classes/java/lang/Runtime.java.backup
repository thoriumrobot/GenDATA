/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2019, Azul Systems, Inc. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.tainting.qual.Untainted;
import org.checkerframework.dataflow.qual.TerminatesExecution;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.math.BigInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.List;
import java.util.Optional;
import java.util.StringTokenizer;
import jdk.internal.access.SharedSecrets;
import jdk.internal.loader.NativeLibrary;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;

@AnnotatedFor({ "interning", "nullness", "tainting" })
@UsesObjectEquals
public class Runtime {

    public static Runtime getRuntime();

    @TerminatesExecution
    public void exit(int status);

    public void addShutdownHook(Thread hook);

    public boolean removeShutdownHook(Thread hook);

    public void halt(int status);

    public Process exec(@Untainted String command) throws IOException;

    public Process exec(@Untainted String command, @Untainted String @Nullable [] envp) throws IOException;

    public Process exec(@Untainted String command, @Untainted String @Nullable [] envp, @Nullable File dir) throws IOException;

    public Process exec(@Untainted String[] cmdarray) throws IOException;

    public Process exec(@Untainted String[] cmdarray, @Untainted String @Nullable [] envp) throws IOException;

    public Process exec(@Untainted String[] cmdarray, @Untainted String @Nullable [] envp, @Nullable File dir) throws IOException;

    public native int availableProcessors();

    public native long freeMemory();

    public native long totalMemory();

    public native long maxMemory();

    public native void gc();

    public void runFinalization();

    @CallerSensitive
    public void load(String filename);

    void load0(Class<?> fromClass, String filename);

    @CallerSensitive
    public void loadLibrary(String libname);

    void loadLibrary0(Class<?> fromClass, String libname);

    public static Version version();

    @jdk.internal.ValueBased
    public static final class Version implements Comparable<Version> {

        public static Version parse(String s);

        public int feature();

        public int interim();

        public int update();

        public int patch();

        @Deprecated()
        public int major();

        @Deprecated()
        public int minor();

        @Deprecated()
        public int security();

        public List<Integer> version();

        public Optional<String> pre();

        public Optional<Integer> build();

        public Optional<String> optional();

        @Override
        public int compareTo(Version obj);

        public int compareToIgnoreOptional(Version obj);

        @Override
        public String toString();

        @Override
        public boolean equals(Object obj);

        public boolean equalsIgnoreOptional(Object obj);

        @Override
        public int hashCode();
    }

    private static class VersionPattern {
    }
}
