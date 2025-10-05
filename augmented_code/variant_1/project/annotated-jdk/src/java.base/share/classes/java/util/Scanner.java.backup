/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.mustcall.qual.MustCall;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.common.returnsreceiver.qual.This;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.math.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.charset.*;
import java.nio.file.Path;
import java.nio.file.Files;
import java.text.*;
import java.text.spi.NumberFormatProvider;
import java.util.function.Consumer;
import java.util.regex.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.ResourceBundleBasedAdapter;

@AnnotatedFor({ "index", "interning", "lock", "mustcall", "nullness", "signedness" })
@UsesObjectEquals
public final class Scanner implements Iterator<String>, Closeable {

    @MustCallAlias
    public Scanner(@MustCallAlias Readable source) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias InputStream source) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias InputStream source, String charsetName) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias InputStream source, Charset charset) {
    }

    public Scanner(File source) throws FileNotFoundException {
    }

    public Scanner(File source, String charsetName) throws FileNotFoundException {
    }

    public Scanner(File source, Charset charset) throws IOException {
    }

    public Scanner(Path source) throws IOException {
    }

    public Scanner(Path source, String charsetName) throws IOException {
    }

    public Scanner(Path source, Charset charset) throws IOException {
    }

    @MustCall({})
    public Scanner(String source) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias ReadableByteChannel source) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias ReadableByteChannel source, String charsetName) {
    }

    @MustCallAlias
    public Scanner(@MustCallAlias ReadableByteChannel source, Charset charset) {
    }

    public void close();

    @Nullable
    public IOException ioException();

    public Pattern delimiter();

    @This
    public Scanner useDelimiter(Pattern pattern);

    @This
    public Scanner useDelimiter(String pattern);

    public Locale locale();

    @This
    public Scanner useLocale(Locale locale);

    @Positive
    @IntRange(from = 2, to = 36)
    public int radix();

    @This
    public Scanner useRadix(@IntRange(from = 2, to = 36) int radix);

    public MatchResult match();

    @SideEffectFree
    public String toString(@GuardSatisfied Scanner this);

    @Pure
    @EnsuresNonEmptyIf(result = true, expression = "this")
    public boolean hasNext(@GuardSatisfied Scanner this);

    @SideEffectsOnly("this")
    public String next(@GuardSatisfied @NonEmpty Scanner this);

    public void remove(@GuardSatisfied Scanner this);

    @Pure
    @EnsuresNonEmptyIf(result = true, expression = "this")
    public boolean hasNext(@GuardSatisfied Scanner this, String pattern);

    @SideEffectsOnly("this")
    public String next(@GuardSatisfied @NonEmpty Scanner this, String pattern);

    @Pure
    @EnsuresNonEmptyIf(result = true, expression = "this")
    public boolean hasNext(@GuardSatisfied Scanner this, Pattern pattern);

    @SideEffectsOnly("this")
    public String next(@GuardSatisfied @NonEmpty Scanner this, Pattern pattern);

    @Pure
    public boolean hasNextLine();

    @SideEffectsOnly("this")
    public String nextLine(@GuardSatisfied @NonEmpty Scanner this);

    @Nullable
    public String findInLine(String pattern);

    @Nullable
    public String findInLine(Pattern pattern);

    @Nullable
    public String findWithinHorizon(String pattern, @NonNegative int horizon);

    @Nullable
    public String findWithinHorizon(Pattern pattern, @NonNegative int horizon);

    @This
    public Scanner skip(@GuardSatisfied @NonEmpty Scanner this, Pattern pattern);

    @This
    public Scanner skip(@GuardSatisfied Scanner this, String pattern);

    @Pure
    public boolean hasNextBoolean(@GuardSatisfied Scanner this);

    @SideEffectsOnly("this")
    public boolean nextBoolean(@GuardSatisfied @NonEmpty Scanner this);

    @Pure
    public boolean hasNextByte(@GuardSatisfied Scanner this);

    @Pure
    public boolean hasNextByte(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectsOnly("this")
    @PolySigned
    public byte nextByte(@GuardSatisfied @NonEmpty Scanner this);

    @SideEffectsOnly("this")
    @PolySigned
    public byte nextByte(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    public boolean hasNextShort(@GuardSatisfied Scanner this);

    @Pure
    public boolean hasNextShort(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectsOnly("this")
    @PolySigned
    public short nextShort(@GuardSatisfied @NonEmpty Scanner this);

    @SideEffectsOnly("this")
    @PolySigned
    public short nextShort(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    public boolean hasNextInt(@GuardSatisfied Scanner this);

    @Pure
    public boolean hasNextInt(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectsOnly("this")
    @PolySigned
    public int nextInt(@GuardSatisfied @NonEmpty Scanner this);

    @SideEffectsOnly("this")
    @PolySigned
    public int nextInt(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    public boolean hasNextLong(@GuardSatisfied Scanner this);

    @Pure
    public boolean hasNextLong(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @SideEffectsOnly("this")
    @PolySigned
    public long nextLong(@GuardSatisfied @NonEmpty Scanner this);

    @SideEffectsOnly("this")
    @PolySigned
    public long nextLong(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    public boolean hasNextFloat(@GuardSatisfied Scanner this);

    @SideEffectsOnly("this")
    public float nextFloat(@GuardSatisfied @NonEmpty Scanner this);

    @Pure
    public boolean hasNextDouble(@GuardSatisfied Scanner this);

    @SideEffectsOnly("this")
    public double nextDouble(@GuardSatisfied @NonEmpty Scanner this);

    @Pure
    public boolean hasNextBigInteger(@GuardSatisfied Scanner this);

    @Pure
    public boolean hasNextBigInteger(@GuardSatisfied Scanner this, @IntRange(from = 2, to = 36) int radix);

    @SideEffectsOnly("this")
    public BigInteger nextBigInteger(@GuardSatisfied @NonEmpty Scanner this);

    @SideEffectsOnly("this")
    public BigInteger nextBigInteger(@GuardSatisfied @NonEmpty Scanner this, @IntRange(from = 2, to = 36) int radix);

    @Pure
    public boolean hasNextBigDecimal(@GuardSatisfied Scanner this);

    @SideEffectsOnly("this")
    public BigDecimal nextBigDecimal(@GuardSatisfied @NonEmpty Scanner this);

    @This
    public Scanner reset(@GuardSatisfied Scanner this);

    public Stream<String> tokens();

    class TokenSpliterator extends Spliterators.AbstractSpliterator<String> {

        @Override
        public boolean tryAdvance(Consumer<? super String> cons);
    }

    public Stream<MatchResult> findAll(Pattern pattern);

    public Stream<MatchResult> findAll(String patString);

    class FindSpliterator extends Spliterators.AbstractSpliterator<MatchResult> {

        @Override
        public boolean tryAdvance(Consumer<? super MatchResult> cons);
    }

    private static class PatternLRUCache {

        boolean hasName(Pattern p, String s);

        void moveToFront(Object[] oa, int i);

        Pattern forName(String name);
    }
}
