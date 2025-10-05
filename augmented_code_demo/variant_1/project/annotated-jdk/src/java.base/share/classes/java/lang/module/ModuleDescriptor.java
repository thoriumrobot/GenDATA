/*
    @Positive
 * Copyright (c) 2009, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.module;

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
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.file.Path;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import static jdk.internal.module.Checks.*;
    @Positive
import static java.util.Objects.*;
    @Positive
import jdk.internal.module.Checks;
    @Positive
import jdk.internal.module.ModuleInfo;

    @Positive
public class ModuleDescriptor implements Comparable<ModuleDescriptor> {

    @Positive
    public enum Modifier {

    @Positive
        OPEN, AUTOMATIC, SYNTHETIC, MANDATED
    @Positive
    }

    @Positive
    public static final class Requires implements Comparable<Requires> {

    @Positive
        public enum Modifier {

    @Positive
            TRANSITIVE, STATIC, SYNTHETIC, MANDATED
    @Positive
        }

    @Positive
        public Set<Modifier> modifiers();

    @Positive
        public String name();

    @Positive
        public Optional<Version> compiledVersion();

    @Positive
        public Optional<String> rawCompiledVersion();

    @Positive
        @Override
    @Positive
        public int compareTo(Requires that);

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class Exports implements Comparable<Exports> {

    @Positive
        public enum Modifier {

    @Positive
            SYNTHETIC, MANDATED
    @Positive
        }

    @Positive
        public Set<Modifier> modifiers();

    @Positive
        public boolean isQualified();

    @Positive
        public String source();

    @Positive
        public Set<String> targets();

    @Positive
        @Override
    @Positive
        public int compareTo(Exports that);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class Opens implements Comparable<Opens> {

    @Positive
        public enum Modifier {

    @Positive
            SYNTHETIC, MANDATED
    @Positive
        }

    @Positive
        public Set<Modifier> modifiers();

    @Positive
        public boolean isQualified();

    @Positive
        public String source();

    @Positive
        public Set<String> targets();

    @Positive
        @Override
    @Positive
        public int compareTo(Opens that);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class Provides implements Comparable<Provides> {

    @Positive
        public String service();

    @Positive
        public List<String> providers();

    @Positive
        public int compareTo(Provides that);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static final class Version implements Comparable<Version> {

    @Positive
        public static Version parse(String v);

    @Positive
        @Override
    @Positive
        public int compareTo(Version that);

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public String name();

    @Positive
    public Set<Modifier> modifiers();

    @Positive
    public boolean isOpen();

    @Positive
    public boolean isAutomatic();

    @Positive
    public Set<Requires> requires();

    @Positive
    public Set<Exports> exports();

    @Positive
    public Set<Opens> opens();

    @Positive
    public Set<String> uses();

    @Positive
    public Set<Provides> provides();

    @Positive
    public Optional<Version> version();

    @Positive
    public Optional<String> rawVersion();

    @Positive
    public String toNameAndVersion();

    @Positive
    public Optional<String> mainClass();

    @Positive
    public Set<String> packages();

    @Positive
    public static final class Builder {

    @Positive
        Set<String> packages();

    @Positive
        public Builder requires(Requires req);

    @Positive
        public Builder requires(Set<Requires.Modifier> ms, String mn, Version compiledVersion);

    @Positive
        Builder requires(Set<Requires.Modifier> ms, String mn, String rawCompiledVersion);

    @Positive
        public Builder requires(Set<Requires.Modifier> ms, String mn);

    @Positive
        public Builder requires(String mn);

    @Positive
        public Builder exports(Exports e);

    @Positive
        public Builder exports(Set<Exports.Modifier> ms, String pn, Set<String> targets);

    @Positive
        public Builder exports(Set<Exports.Modifier> ms, String pn);

    @Positive
        public Builder exports(String pn, Set<String> targets);

    @Positive
        public Builder exports(String pn);

    @Positive
        public Builder opens(Opens obj);

    @Positive
        public Builder opens(Set<Opens.Modifier> ms, String pn, Set<String> targets);

    @Positive
        public Builder opens(Set<Opens.Modifier> ms, String pn);

    @Positive
        public Builder opens(String pn, Set<String> targets);

    @Positive
        public Builder opens(String pn);

    @Positive
        public Builder uses(String service);

    @Positive
        public Builder provides(Provides p);

    @Positive
        public Builder provides(String service, List<String> providers);

    @Positive
        public Builder packages(Set<String> pns);

    @Positive
        public Builder version(Version v);

    @Positive
        public Builder version(String vs);

    @Positive
        public Builder mainClass(String mc);

    @Positive
        public ModuleDescriptor build();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public int compareTo(ModuleDescriptor that);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public static Builder newModule(String name, Set<Modifier> ms);

    @Positive
    public static Builder newModule(String name);

    @Positive
    public static Builder newOpenModule(String name);

    @Positive
    public static Builder newAutomaticModule(String name);

    @Positive
    public static ModuleDescriptor read(InputStream in, Supplier<Set<String>> packageFinder) throws IOException;

    @Positive
    public static ModuleDescriptor read(InputStream in) throws IOException;

    @Positive
    public static ModuleDescriptor read(ByteBuffer bb, Supplier<Set<String>> packageFinder);

    @Positive
    public static ModuleDescriptor read(ByteBuffer bb);
    @Positive
}

// CFWR semantic augmentation - variant 1
