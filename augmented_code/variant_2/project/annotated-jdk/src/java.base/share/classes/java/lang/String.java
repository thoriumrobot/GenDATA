/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.formatter.qual.FormatMethod;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.checker.index.qual.SubstringIndexFor;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.regex.qual.PolyRegex;
    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.checker.signature.qual.PolySignature;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.common.aliasing.qual.Unique;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.ArrayLenRange;
    @Positive
import org.checkerframework.common.value.qual.EnsuresMinLenIf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.common.value.qual.StringVal;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.lang.annotation.Native;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDesc;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.*;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.Formatter;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.regex.PatternSyntaxException;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.nio.cs.ArrayDecoder;
    @Positive
import sun.nio.cs.ArrayEncoder;
    @Positive
import sun.nio.cs.ISO_8859_1;
    @Positive
import sun.nio.cs.US_ASCII;
    @Positive
import sun.nio.cs.UTF_8;

    @Positive
@AnnotatedFor({ "aliasing", "formatter", "index", "interning", "lock", "nullness", "regex", "signature", "signedness" })
    @Positive
public final class String implements java.io.Serializable, Comparable<String>, CharSequence, Constable, ConstantDesc {

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @StringVal("")
    @Positive
    @Unique
    @Positive
    public String() {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyValue
    @Positive
    @Unique
    @Positive
    public String(@PolyValue String original) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    @Unique
    @Positive
    public String(char @GuardSatisfied @PolyValue [] value) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(char @GuardSatisfied [] value, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int count) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(int @GuardSatisfied [] codePoints, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int count) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @Unique
    @Positive
    public String(byte @GuardSatisfied [] ascii, int hibyte, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int count) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @Unique
    @Positive
    public String(byte @GuardSatisfied [] ascii, int hibyte) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int length, String charsetName) throws UnsupportedEncodingException {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int length, Charset charset) {
    @Positive
    }

    @Positive
    static String newStringUTF8NoRepl(byte[] bytes, int offset, int length);

    @Positive
    static String newStringNoRepl(byte[] src, Charset cs) throws CharacterCodingException;

    @Positive
    static byte[] getBytesUTF8NoRepl(String s);

    @Positive
    static byte[] getBytesNoRepl(String s, Charset cs) throws CharacterCodingException;

    @Positive
    static int decodeASCII(byte[] sa, int sp, char[] da, int dp, int len);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes, String charsetName) throws UnsupportedEncodingException {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes, Charset charset) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int length) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@PolySigned byte @GuardSatisfied [] bytes) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@GuardSatisfied StringBuffer buffer) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @Unique
    @Positive
    public String(@GuardSatisfied StringBuilder builder) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @LengthOf({ "this" })
    @Positive
    public int length();

    @Positive
    @SuppressWarnings("contracts.conditional.postcondition.not.satisfied")
    @Positive
    @CFComment("index: The postcondition is EnsuresMinLenIf.  It's true because: values.length != 0 => this is @MinLen(1), as values.length is @LengthOf(this).")
    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @EnsuresMinLenIf(expression = "this", result = false, targetValue = 1)
    @Positive
    @Override
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public char charAt(@IndexFor({ "this" }) int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int codePointAt(@IndexFor({ "this" }) int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int codePointBefore(@LTEqLengthOf({ "this" }) @Positive int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NonNegative
    @Positive
    public int codePointCount(@IndexOrHigh({ "this" }) int beginIndex, @IndexOrHigh({ "this" }) int endIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrHigh({ "this" })
    @Positive
    public int offsetByCodePoints(@IndexOrHigh({ "this" }) int index, int codePointOffset);

    @Positive
    public void getChars(@IndexOrHigh({ "this" }) int srcBegin, @IndexOrHigh({ "this" }) int srcEnd, char @GuardSatisfied [] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    @Positive
    @Deprecated()
    @Positive
    public void getBytes(@IndexOrHigh({ "this" }) int srcBegin, @IndexOrHigh({ "this" }) int srcEnd, byte @GuardSatisfied [] dst, @IndexOrHigh({ "#3" }) int dstBegin);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public byte[] getBytes(String charsetName) throws UnsupportedEncodingException;

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public byte[] getBytes(Charset charset);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public byte[] getBytes();

    @Positive
    @EnsuresNonNullIf(expression = { "#1" }, result = true)
    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equals(@GuardSatisfied @Nullable Object anObject);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean contentEquals(@GuardSatisfied StringBuffer sb);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean contentEquals(@GuardSatisfied CharSequence cs);

    @Positive
    @EnsuresNonNullIf(expression = { "#1" }, result = true)
    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equalsIgnoreCase(@Nullable String anotherString);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(String anotherString);

    @Positive
    public static final Comparator<String> CASE_INSENSITIVE_ORDER;

    @Positive
    private static class CaseInsensitiveComparator implements Comparator<String>, java.io.Serializable {

    @Positive
        public int compare(String s1, String s2);
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareToIgnoreCase(String str);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean regionMatches(int toffset, String other, int ooffset, int len);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean regionMatches(boolean ignoreCase, int toffset, String other, int ooffset, int len);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean startsWith(String prefix, int toffset);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean startsWith(String prefix);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean endsWith(String suffix);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrLow({ "this" })
    @Positive
    public int indexOf(int ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrLow({ "this" })
    @Positive
    public int indexOf(int ch, int fromIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrLow({ "this" })
    @Positive
    public int lastIndexOf(int ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrLow({ "this" })
    @Positive
    public int lastIndexOf(int ch, int fromIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @LTEqLengthOf({ "this" })
    @Positive
    @SubstringIndexFor(value = { "this" }, offset = { "#1.length()-1" })
    @Positive
    public int indexOf(String str);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @LTEqLengthOf({ "this" })
    @Positive
    @SubstringIndexFor(value = { "this" }, offset = { "#1.length()-1" })
    @Positive
    public int indexOf(String str, int fromIndex);

    @Positive
    static int indexOf(byte[] src, byte srcCoder, int srcCount, String tgtStr, int fromIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @LTEqLengthOf({ "this" })
    @Positive
    @SubstringIndexFor(value = { "this" }, offset = { "#1.length()-1" })
    @Positive
    public int lastIndexOf(String str);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @LTEqLengthOf({ "this" })
    @Positive
    @SubstringIndexFor(value = { "this" }, offset = { "#1.length()-1" })
    @Positive
    public int lastIndexOf(String str, int fromIndex);

    @Positive
    static int lastIndexOf(byte[] src, byte srcCoder, int srcCount, String tgtStr, int fromIndex);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String substring(@IndexOrHigh({ "this" }) int beginIndex);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String substring(@IndexOrHigh({ "this" }) int beginIndex, @IndexOrHigh({ "this" }) int endIndex);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public CharSequence subSequence(@IndexOrHigh({ "this" }) int beginIndex, @IndexOrHigh({ "this" }) int endIndex);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String concat(String str);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String replace(char oldChar, char newChar);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean matches(@Regex String regex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(CharSequence s);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String replaceFirst(@Regex String regex, String replacement);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String replaceAll(@Regex String regex, String replacement);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String replace(@GuardSatisfied CharSequence target, @GuardSatisfied CharSequence replacement);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String @MinLen(1) [] split(@Regex String regex, int limit);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String @MinLen(1) [] split(@Regex String regex);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String join(CharSequence delimiter, CharSequence... elements);

    @Positive
    @ForceInline
    @Positive
    static String join(String prefix, String suffix, String delimiter, String[] elements, int size);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String join(CharSequence delimiter, Iterable<? extends CharSequence> elements);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String toLowerCase(@GuardSatisfied Locale locale);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String toLowerCase();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String toUpperCase(@GuardSatisfied Locale locale);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String toUpperCase();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String trim();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String strip();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String stripLeading();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String stripTrailing();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean isBlank();

    @Positive
    public Stream<String> lines();

    @Positive
    @CFComment("n may be negative")
    @Positive
    @SideEffectFree
    @Positive
    public String indent(int n);

    @Positive
    @SideEffectFree
    @Positive
    public String stripIndent();

    @Positive
    @SideEffectFree
    @Positive
    public String translateEscapes();

    @Positive
    public <R> R transform(Function<? super String, ? extends R> f);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @SameLen({ "this" })
    @Positive
    @PolyRegex
    @Positive
    @PolyValue
    @Positive
    public String toString(@PolyRegex @PolyValue String this);

    @Positive
    @Override
    @Positive
    public IntStream chars();

    @Positive
    @Override
    @Positive
    public IntStream codePoints();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @PolySigned
    @Positive
    public char @SameLen({ "this" }) @PolyValue [] toCharArray(@PolyValue String this);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @FormatMethod
    @Positive
    public static String format(String format, @GuardSatisfied @Nullable Object@GuardSatisfied ... args);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @FormatMethod
    @Positive
    public static String format(@GuardSatisfied @Nullable Locale l, String format, @GuardSatisfied @Nullable Object@GuardSatisfied ... args);

    @Positive
    @SideEffectFree
    @Positive
    public String formatted(Object... args);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static String valueOf(@GuardSatisfied @Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    @SameLen({ "#1" })
    @Positive
    @PolyValue
    @Positive
    public static String valueOf(char @GuardSatisfied @PolyValue [] data);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static String valueOf(char @GuardSatisfied [] data, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int count);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public static String copyValueOf(char @GuardSatisfied [] data, @IndexOrHigh({ "#1" }) int offset, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int count);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @SameLen({ "#1" })
    @Positive
    @PolyValue
    @Positive
    public static String copyValueOf(char @GuardSatisfied @PolyValue [] data);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    @StringVal({ "true", "false" })
    @Positive
    public static String valueOf(boolean b);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    @ArrayLen(1)
    @Positive
    public static String valueOf(char c);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    @ArrayLenRange(from = 1, to = 11)
    @Positive
    public static String valueOf(int i);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    @ArrayLenRange(from = 1, to = 20)
    @Positive
    public static String valueOf(long l);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static String valueOf(float f);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @NewObject
    @Positive
    public static String valueOf(double d);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Interned
    @Positive
    @SameLen({ "this" })
    @Positive
    @PolyRegex
    @Positive
    @PolySignature
    @Positive
    @PolyValue
    @Positive
    public native String intern(@PolyRegex @PolySignature @PolyValue String this);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    public String repeat(int count);

    @Positive
    void getBytes(byte[] dst, int dstBegin, byte coder);

    @Positive
    void getBytes(byte[] dst, int srcPos, int dstBegin, byte coder, int length);

    @Positive
    byte coder();

    @Positive
    byte[] value();

    @Positive
    boolean isLatin1();

    @Positive
    static void checkIndex(int index, int length);

    @Positive
    static void checkOffset(int offset, int length);

    @Positive
    static void checkBoundsOffCount(int offset, int count, int length);

    @Positive
    static void checkBoundsBeginEnd(int begin, int end, int length);

    @Positive
    static String valueOfCodePoint(int codePoint);

    @Positive
    @Override
    @Positive
    public Optional<String> describeConstable();

    @Positive
    @Override
    @Positive
    public String resolveConstantDesc(MethodHandles.Lookup lookup);
    @Positive
}
