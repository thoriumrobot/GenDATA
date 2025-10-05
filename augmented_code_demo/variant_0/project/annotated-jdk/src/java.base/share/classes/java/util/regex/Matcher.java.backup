/*
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.regex;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public final class Matcher implements MatchResult {

    @Pure
    public Pattern pattern();

    @SideEffectFree
    public MatchResult toMatchResult();

    private static class ImmutableMatchResult implements MatchResult {

        @Override
        @Pure
        public int start();

        @Override
        @Pure
        public int start(int group);

        @Override
        @Pure
        public int end();

        @Override
        @Pure
        public int end(int group);

        @Override
        @Pure
        public int groupCount();

        @Override
        @SideEffectFree
        public String group();

        @Override
        @SideEffectFree
        public String group(int group);
    }

    public Matcher usePattern(Pattern newPattern);

    public Matcher reset();

    public Matcher reset(CharSequence input);

    @Pure
    @NonNegative
    public int start();

    @Pure
    @GTENegativeOne
    public int start(@NonNegative int group);

    @Pure
    public int start(String name);

    @Pure
    @NonNegative
    public int end();

    @Pure
    @GTENegativeOne
    public int end(@NonNegative int group);

    @Pure
    public int end(String name);

    @SideEffectFree
    public String group();

    @SideEffectFree
    @Nullable
    public String group(@NonNegative int group);

    @SideEffectFree
    @Nullable
    public String group(String name);

    @Pure
    @NonNegative
    public int groupCount();

    @Pure
    public boolean matches();

    public boolean find();

    public boolean find(@NonNegative int start);

    @Pure
    public boolean lookingAt();

    @SideEffectFree
    public static String quoteReplacement(String s);

    public Matcher appendReplacement(StringBuffer sb, String replacement);

    public Matcher appendReplacement(StringBuilder sb, String replacement);

    public StringBuffer appendTail(StringBuffer sb);

    public StringBuilder appendTail(StringBuilder sb);

    @SideEffectFree
    public String replaceAll(String replacement);

    @SideEffectFree
    public String replaceAll(Function<MatchResult, String> replacer);

    public Stream<MatchResult> results();

    public String replaceFirst(String replacement);

    public String replaceFirst(Function<MatchResult, String> replacer);

    public Matcher region(@NonNegative int start, @NonNegative int end);

    @Pure
    @NonNegative
    public int regionStart();

    @Pure
    @NonNegative
    public int regionEnd();

    @Pure
    public boolean hasTransparentBounds();

    public Matcher useTransparentBounds(boolean b);

    @Pure
    public boolean hasAnchoringBounds();

    public Matcher useAnchoringBounds(boolean b);

    @SideEffectFree
    public String toString(@GuardSatisfied Matcher this);

    @Pure
    public boolean hitEnd();

    @Pure
    public boolean requireEnd();

    boolean search(int from);

    boolean match(int from, int anchor);

    int getTextLength();

    CharSequence getSubSequence(int beginIndex, int endIndex);

    char charAt(int i);

    int getMatchedGroupIndex(String name);
}
