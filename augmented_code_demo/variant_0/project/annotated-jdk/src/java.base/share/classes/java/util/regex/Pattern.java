/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.regex.qual.PolyRegex;
    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.text.Normalizer;
    @Positive
import java.text.Normalizer.Form;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.util.ArraysSupport;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness", "regex" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Pattern implements java.io.Serializable {

    @Positive
    @SignedPositive
    @Positive
    public static final int UNIX_LINES;

    @Positive
    @SignedPositive
    @Positive
    public static final int CASE_INSENSITIVE;

    @Positive
    @SignedPositive
    @Positive
    public static final int COMMENTS;

    @Positive
    @SignedPositive
    @Positive
    public static final int MULTILINE;

    @Positive
    @SignedPositive
    @Positive
    public static final int LITERAL;

    @Positive
    @SignedPositive
    @Positive
    public static final int DOTALL;

    @Positive
    @SignedPositive
    @Positive
    public static final int UNICODE_CASE;

    @Positive
    @SignedPositive
    @Positive
    public static final int CANON_EQ;

    @Positive
    @SignedPositive
    @Positive
    public static final int UNICODE_CHARACTER_CLASS;

    @Positive
    @CFComment({ "lock/nullness: pure wrt equals(@GuardSatisfied Pattern this) but not ==" })
    @Positive
    @Pure
    @Positive
    public static Pattern compile(@Regex String regex);

    @Positive
    @CFComment({ "lock/nullness: pure wrt equals(@GuardSatisfied Pattern this) but not ==" })
    @Positive
    @Pure
    @Positive
    public static Pattern compile(@Regex String regex, int flags);

    @Positive
    @Pure
    @Positive
    public String pattern();

    @Positive
    @Pure
    @Positive
    public String toString(@GuardSatisfied Pattern this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyRegex
    @Positive
    public Matcher matcher(@PolyRegex Pattern this, CharSequence input);

    @Positive
    @Pure
    @Positive
    public int flags();

    @Positive
    @Pure
    @Positive
    public static boolean matches(@Regex String regex, CharSequence input);

    @Positive
    @Pure
    @Positive
    public String @MinLen(1) [] split(CharSequence input, int limit);

    @Positive
    @Pure
    @Positive
    public String @MinLen(1) [] split(CharSequence input);

    @Positive
    @CFComment({ "nullness: pure wrt equals() but not ==" })
    @Positive
    @Pure
    @Positive
    @Regex
    @Positive
    public static String quote(String s);

    @Positive
    Map<String, Integer> namedGroups();

    @Positive
    static final class TreeInfo {

    @Positive
        void reset();
    @Positive
    }

    @Positive
    static final class BitClass implements BmpCharPredicate {

    @Positive
        BitClass add(int c, int flags);

    @Positive
        public boolean is(int ch);
    @Positive
    }

    @Positive
    static class Node extends Object {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class LastNode extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class Start extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class StartS extends Start {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Begin extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class End extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Caret extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class UnixCaret extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class LastMatch extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Dollar extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class UnixDollar extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class LineEnding extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class CharProperty extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    private static class BmpCharProperty extends CharProperty {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    private static class NFCCharProperty extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class XGrapheme extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class GraphemeBound extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class SliceNode extends Node {

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class Slice extends SliceNode {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class SliceI extends SliceNode {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class SliceU extends SliceNode {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class SliceS extends Slice {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class SliceIS extends SliceNode {

    @Positive
        int toLower(int c);

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class SliceUS extends SliceIS {

    @Positive
        int toLower(int c);
    @Positive
    }

    @Positive
    static final class Ques extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class CharPropertyGreedy extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class BmpCharPropertyGreedy extends CharPropertyGreedy {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Curly extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean match0(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean match1(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean match2(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class GroupCurly extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean match0(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean match1(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean match2(Matcher matcher, int i, int j, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class BranchConn extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class Branch extends Node {

    @Positive
        void add(Node node);

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class GroupHead extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class GroupTail extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Prolog extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class Loop extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean matchInit(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class LazyLoop extends Loop {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean matchInit(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class BackRef extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static class CIBackRef extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class First extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class Pos extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Neg extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class LookBehindEndNode extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class Behind extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class BehindS extends Behind {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class NotBehind extends Node {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class NotBehindS extends NotBehind {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static final class Bound extends Node {

    @Positive
        boolean isWord(int ch);

    @Positive
        int check(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    static class BnM extends Node {

    @Positive
        static Node optimize(Node node);

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);

    @Positive
        boolean study(TreeInfo info);
    @Positive
    }

    @Positive
    static final class BnMS extends BnM {

    @Positive
        boolean match(Matcher matcher, int i, CharSequence seq);
    @Positive
    }

    @Positive
    @FunctionalInterface
    @Positive
    static interface CharPredicate {

    @Positive
        boolean is(int ch);

    @Positive
        default CharPredicate and(CharPredicate p);

    @Positive
        default CharPredicate union(CharPredicate p);

    @Positive
        default CharPredicate union(CharPredicate p1, CharPredicate p2);

    @Positive
        default CharPredicate negate();
    @Positive
    }

    @Positive
    static interface BmpCharPredicate extends CharPredicate {

    @Positive
        default CharPredicate and(CharPredicate p);

    @Positive
        default CharPredicate union(CharPredicate p);

    @Positive
        static CharPredicate union(CharPredicate... predicates);
    @Positive
    }

    @Positive
    static BmpCharPredicate VertWS();

    @Positive
    static BmpCharPredicate HorizWS();

    @Positive
    static CharPredicate ALL();

    @Positive
    static CharPredicate DOT();

    @Positive
    static CharPredicate UNIXDOT();

    @Positive
    static CharPredicate SingleS(int c);

    @Positive
    static BmpCharPredicate Single(int c);

    @Positive
    static BmpCharPredicate SingleI(int lower, int upper);

    @Positive
    static CharPredicate SingleU(int lower);

    @Positive
    static CharPredicate Range(int lower, int upper);

    @Positive
    static CharPredicate CIRange(int lower, int upper);

    @Positive
    static CharPredicate CIRangeU(int lower, int upper);

    @Positive
    @SideEffectFree
    @Positive
    public Predicate<String> asPredicate();

    @Positive
    @SideEffectFree
    @Positive
    public Predicate<String> asMatchPredicate();

    @Positive
    @SideEffectFree
    @Positive
    public Stream<String> splitAsStream(final CharSequence input);
    @Positive
}

// CFWR semantic augmentation - variant 0
