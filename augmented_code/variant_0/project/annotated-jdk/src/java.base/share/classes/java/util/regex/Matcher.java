/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.regex;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ConcurrentModificationException;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Matcher implements MatchResult {

    @Positive
    @Pure
    @Positive
    public Pattern pattern();

    @Positive
    @SideEffectFree
    @Positive
    public MatchResult toMatchResult();

    @Positive
    private static class ImmutableMatchResult implements MatchResult {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int start();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int start(int group);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int end();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int end(int group);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int groupCount();

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public String group();

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public String group(int group);
    @Positive
    }

    @Positive
    public Matcher usePattern(Pattern newPattern);

    @Positive
    public Matcher reset();

    @Positive
    public Matcher reset(CharSequence input);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int start();

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int start(@NonNegative int group);

    @Positive
    @Pure
    @Positive
    public int start(String name);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int end();

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int end(@NonNegative int group);

    @Positive
    @Pure
    @Positive
    public int end(String name);

    @Positive
    @SideEffectFree
    @Positive
    public String group();

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public String group(@NonNegative int group);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public String group(String name);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int groupCount();

    @Positive
    @Pure
    @Positive
    public boolean matches();

    @Positive
    public boolean find();

    @Positive
    public boolean find(@NonNegative int start);

    @Positive
    @Pure
    @Positive
    public boolean lookingAt();

    @Positive
    @SideEffectFree
    @Positive
    public static String quoteReplacement(String s);

    @Positive
    public Matcher appendReplacement(StringBuffer sb, String replacement);

    @Positive
    public Matcher appendReplacement(StringBuilder sb, String replacement);

    @Positive
    public StringBuffer appendTail(StringBuffer sb);

    @Positive
    public StringBuilder appendTail(StringBuilder sb);

    @Positive
    @SideEffectFree
    @Positive
    public String replaceAll(String replacement);

    @Positive
    @SideEffectFree
    @Positive
    public String replaceAll(Function<MatchResult, String> replacer);

    @Positive
    public Stream<MatchResult> results();

    @Positive
    public String replaceFirst(String replacement);

    @Positive
    public String replaceFirst(Function<MatchResult, String> replacer);

    @Positive
    public Matcher region(@NonNegative int start, @NonNegative int end);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int regionStart();

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int regionEnd();

    @Positive
    @Pure
    @Positive
    public boolean hasTransparentBounds();

    @Positive
    public Matcher useTransparentBounds(boolean b);

    @Positive
    @Pure
    @Positive
    public boolean hasAnchoringBounds();

    @Positive
    public Matcher useAnchoringBounds(boolean b);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Matcher this);

    @Positive
    @Pure
    @Positive
    public boolean hitEnd();

    @Positive
    @Pure
    @Positive
    public boolean requireEnd();

    @Positive
    boolean search(int from);

    @Positive
    boolean match(int from, int anchor);

    @Positive
    int getTextLength();

    @Positive
    CharSequence getSubSequence(int beginIndex, int endIndex);

    @Positive
    char charAt(int i);

    @Positive
    int getMatchedGroupIndex(String name);
    @Positive
}
