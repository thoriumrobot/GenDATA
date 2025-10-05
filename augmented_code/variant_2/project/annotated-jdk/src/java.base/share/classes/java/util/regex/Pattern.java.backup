/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.regex.qual.PolyRegex;
import org.checkerframework.checker.regex.qual.Regex;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.Locale;
import java.util.Iterator;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.Predicate;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import jdk.internal.util.ArraysSupport;

@AnnotatedFor({ "index", "interning", "lock", "nullness", "regex" })
@UsesObjectEquals
public final class Pattern implements java.io.Serializable {

    @SignedPositive
    public static final int UNIX_LINES;

    @SignedPositive
    public static final int CASE_INSENSITIVE;

    @SignedPositive
    public static final int COMMENTS;

    @SignedPositive
    public static final int MULTILINE;

    @SignedPositive
    public static final int LITERAL;

    @SignedPositive
    public static final int DOTALL;

    @SignedPositive
    public static final int UNICODE_CASE;

    @SignedPositive
    public static final int CANON_EQ;

    @SignedPositive
    public static final int UNICODE_CHARACTER_CLASS;

    @CFComment({ "lock/nullness: pure wrt equals(@GuardSatisfied Pattern this) but not ==" })
    @Pure
    public static Pattern compile(@Regex String regex);

    @CFComment({ "lock/nullness: pure wrt equals(@GuardSatisfied Pattern this) but not ==" })
    @Pure
    public static Pattern compile(@Regex String regex, int flags);

    @Pure
    public String pattern();

    @Pure
    public String toString(@GuardSatisfied Pattern this);

    @SideEffectFree
    @PolyRegex
    public Matcher matcher(@PolyRegex Pattern this, CharSequence input);

    @Pure
    public int flags();

    @Pure
    public static boolean matches(@Regex String regex, CharSequence input);

    @Pure
    public String @MinLen(1) [] split(CharSequence input, int limit);

    @Pure
    public String @MinLen(1) [] split(CharSequence input);

    @CFComment({ "nullness: pure wrt equals() but not ==" })
    @Pure
    @Regex
    public static String quote(String s);

    Map<String, Integer> namedGroups();

    static final class TreeInfo {

        void reset();
    }

    static final class BitClass implements BmpCharPredicate {

        BitClass add(int c, int flags);

        public boolean is(int ch);
    }

    static class Node extends Object {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class LastNode extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class Start extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class StartS extends Start {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Begin extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class End extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Caret extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class UnixCaret extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class LastMatch extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Dollar extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class UnixDollar extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class LineEnding extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class CharProperty extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    private static class BmpCharProperty extends CharProperty {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    private static class NFCCharProperty extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class XGrapheme extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class GraphemeBound extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class SliceNode extends Node {

        boolean study(TreeInfo info);
    }

    static class Slice extends SliceNode {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class SliceI extends SliceNode {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class SliceU extends SliceNode {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class SliceS extends Slice {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class SliceIS extends SliceNode {

        int toLower(int c);

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class SliceUS extends SliceIS {

        int toLower(int c);
    }

    static final class Ques extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class CharPropertyGreedy extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class BmpCharPropertyGreedy extends CharPropertyGreedy {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Curly extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean match0(Matcher matcher, int i, int j, CharSequence seq);

        boolean match1(Matcher matcher, int i, int j, CharSequence seq);

        boolean match2(Matcher matcher, int i, int j, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class GroupCurly extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean match0(Matcher matcher, int i, int j, CharSequence seq);

        boolean match1(Matcher matcher, int i, int j, CharSequence seq);

        boolean match2(Matcher matcher, int i, int j, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class BranchConn extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class Branch extends Node {

        void add(Node node);

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class GroupHead extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class GroupTail extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Prolog extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class Loop extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean matchInit(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class LazyLoop extends Loop {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean matchInit(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class BackRef extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static class CIBackRef extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class First extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class Pos extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Neg extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class LookBehindEndNode extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class Behind extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class BehindS extends Behind {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class NotBehind extends Node {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class NotBehindS extends NotBehind {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static final class Bound extends Node {

        boolean isWord(int ch);

        int check(Matcher matcher, int i, CharSequence seq);

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    static class BnM extends Node {

        static Node optimize(Node node);

        boolean match(Matcher matcher, int i, CharSequence seq);

        boolean study(TreeInfo info);
    }

    static final class BnMS extends BnM {

        boolean match(Matcher matcher, int i, CharSequence seq);
    }

    @FunctionalInterface
    static interface CharPredicate {

        boolean is(int ch);

        default CharPredicate and(CharPredicate p);

        default CharPredicate union(CharPredicate p);

        default CharPredicate union(CharPredicate p1, CharPredicate p2);

        default CharPredicate negate();
    }

    static interface BmpCharPredicate extends CharPredicate {

        default CharPredicate and(CharPredicate p);

        default CharPredicate union(CharPredicate p);

        static CharPredicate union(CharPredicate... predicates);
    }

    static BmpCharPredicate VertWS();

    static BmpCharPredicate HorizWS();

    static CharPredicate ALL();

    static CharPredicate DOT();

    static CharPredicate UNIXDOT();

    static CharPredicate SingleS(int c);

    static BmpCharPredicate Single(int c);

    static BmpCharPredicate SingleI(int lower, int upper);

    static CharPredicate SingleU(int lower);

    static CharPredicate Range(int lower, int upper);

    static CharPredicate CIRange(int lower, int upper);

    static CharPredicate CIRangeU(int lower, int upper);

    @SideEffectFree
    public Predicate<String> asPredicate();

    @SideEffectFree
    public Predicate<String> asMatchPredicate();

    @SideEffectFree
    public Stream<String> splitAsStream(final CharSequence input);
}
