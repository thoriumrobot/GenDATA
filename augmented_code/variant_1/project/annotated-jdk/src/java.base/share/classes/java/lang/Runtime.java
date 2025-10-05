/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * Copyright (c) 2019, Azul Systems, Inc. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.tainting.qual.Untainted;
    @Positive
import org.checkerframework.dataflow.qual.TerminatesExecution;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.List;
    @Positive
import java.util.Optional;
    @Positive
import java.util.StringTokenizer;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.loader.NativeLibrary;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;

    @Positive
@AnnotatedFor({ "interning", "nullness", "tainting" })
    @Positive
@UsesObjectEquals
    @Positive
public class Runtime {

    @Positive
    public static Runtime getRuntime();

    @Positive
    @TerminatesExecution
    @Positive
    public void exit(int status);

    @Positive
    public void addShutdownHook(Thread hook);

    @Positive
    public boolean removeShutdownHook(Thread hook);

    @Positive
    public void halt(int status);

    @Positive
    public Process exec(@Untainted String command) throws IOException;

    @Positive
    public Process exec(@Untainted String command, @Untainted String @Nullable [] envp) throws IOException;

    @Positive
    public Process exec(@Untainted String command, @Untainted String @Nullable [] envp, @Nullable File dir) throws IOException;

    @Positive
    public Process exec(@Untainted String[] cmdarray) throws IOException;

    @Positive
    public Process exec(@Untainted String[] cmdarray, @Untainted String @Nullable [] envp) throws IOException;

    @Positive
    public Process exec(@Untainted String[] cmdarray, @Untainted String @Nullable [] envp, @Nullable File dir) throws IOException;

    @Positive
    public native int availableProcessors();

    @Positive
    public native long freeMemory();

    @Positive
    public native long totalMemory();

    @Positive
    public native long maxMemory();

    @Positive
    public native void gc();

    @Positive
    public void runFinalization();

    @Positive
    @CallerSensitive
    @Positive
    public void load(String filename);

    @Positive
    void load0(Class<?> fromClass, String filename);

    @Positive
    @CallerSensitive
    @Positive
    public void loadLibrary(String libname);

    @Positive
    void loadLibrary0(Class<?> fromClass, String libname);

    @Positive
    public static Version version();

    @Positive
    @jdk.internal.ValueBased
    @Positive
    public static final class Version implements Comparable<Version> {

    @Positive
        public static Version parse(String s);

    @Positive
        public int feature();

    @Positive
        public int interim();

    @Positive
        public int update();

    @Positive
        public int patch();

    @Positive
        @Deprecated()
    @Positive
        public int major();

    @Positive
        @Deprecated()
    @Positive
        public int minor();

    @Positive
        @Deprecated()
    @Positive
        public int security();

    @Positive
        public List<Integer> version();

    @Positive
        public Optional<String> pre();

    @Positive
        public Optional<Integer> build();

    @Positive
        public Optional<String> optional();

    @Positive
        @Override
    @Positive
        public int compareTo(Version obj);

    @Positive
        public int compareToIgnoreOptional(Version obj);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        public boolean equalsIgnoreOptional(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private static class VersionPattern {
    @Positive
    }
    @Positive
}
