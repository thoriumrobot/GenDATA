/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.common.returnsreceiver.qual.This;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.math.*;
    @Positive
import java.nio.*;
    @Positive
import java.nio.channels.*;
    @Positive
import java.nio.charset.*;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Files;
    @Positive
import java.text.*;
    @Positive
import java.text.spi.NumberFormatProvider;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.regex.*;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "mustcall", "nullness", "signedness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Scanner implements Iterator<String>, Closeable {

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias Readable source) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias InputStream source) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias InputStream source, String charsetName) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias InputStream source, Charset charset) {
    @Positive
    }

    @Positive
    public Scanner(File source) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public Scanner(File source, String charsetName) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public Scanner(File source, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public Scanner(Path source) throws IOException {
    @Positive
    }

    @Positive
    public Scanner(Path source, String charsetName) throws IOException {
    @Positive
    }

    @Positive
    public Scanner(Path source, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    @MustCall({})
    @Positive
    public Scanner(String source) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias ReadableByteChannel source) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias ReadableByteChannel source, String charsetName) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Scanner(@MustCallAlias ReadableByteChannel source, Charset charset) {
    @Positive
    }

    @Positive
    public void close();

    @Positive
    @Nullable
    @Positive
    public IOException ioException();

    @Positive
    public Pattern delimiter();

    @Positive
    @This
    @Positive
    public Scanner useDelimiter(Pattern pattern);

    @Positive
    @This
    @Positive
    public Scanner useDelimiter(String pattern);

    @Positive
    public Locale locale();

    @Positive
    @This
    @Positive
    public Scanner useLocale(Locale locale);

    @Positive
    @Positive
    @Positive
    @IntRange(from = 2, to = 36)
    @Positive
    public int radix();

    @Positive
    @This
    @Positive
    public Scanner useRadix(@IntRange(from = 2, to = 36) int radix);

    @Positive
    public MatchResult match();

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean hasNext(@GuardSatisfied Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public String next(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    public void remove(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean hasNext(@GuardSatisfied Scanner this, String pattern);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public String next(@GuardSatisfied @NonEmpty Scanner this, String pattern);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean hasNext(@GuardSatisfied Scanner this, Pattern pattern);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public String next(@GuardSatisfied @NonEmpty Scanner this, Pattern pattern);

    @Positive
    @Pure
    @Positive
    public boolean hasNextLine();

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public String nextLine(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @Nullable
    @Positive
    public String findInLine(String pattern);

    @Positive
    @Nullable
    @Positive
    public String findInLine(Pattern pattern);

    @Positive
    @Nullable
    @Positive
    public String findWithinHorizon(String pattern, @NonNegative int horizon);

    @Positive
    @Nullable
    @Positive
    public String findWithinHorizon(Pattern pattern, @NonNegative int horizon);

    @Positive
    @This
    @Positive
    public Scanner skip(@GuardSatisfied @NonEmpty Scanner this, Pattern pattern);

    @Positive
    @This
    @Positive
    public Scanner skip(@GuardSatisfied Scanner this, String pattern);

    @Positive
    @Pure
    @Positive
    public boolean hasNextBoolean(@GuardSatisfied Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public boolean nextBoolean(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextByte(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextByte(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public byte nextByte(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public byte nextByte(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    public boolean hasNextShort(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextShort(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public short nextShort(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public short nextShort(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    public boolean hasNextInt(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextInt(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public int nextInt(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public int nextInt(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    public boolean hasNextLong(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextLong(@GuardSatisfied Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public long nextLong(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @PolySigned
    @Positive
    public long nextLong(@GuardSatisfied @NonEmpty Scanner this, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    public boolean hasNextFloat(@GuardSatisfied Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public float nextFloat(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextDouble(@GuardSatisfied Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public double nextDouble(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextBigInteger(@GuardSatisfied Scanner this);

    @Positive
    @Pure
    @Positive
    public boolean hasNextBigInteger(@GuardSatisfied Scanner this, @IntRange(from = 2, to = 36) int radix);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public BigInteger nextBigInteger(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public BigInteger nextBigInteger(@GuardSatisfied @NonEmpty Scanner this, @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    public boolean hasNextBigDecimal(@GuardSatisfied Scanner this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public BigDecimal nextBigDecimal(@GuardSatisfied @NonEmpty Scanner this);

    @Positive
    @This
    @Positive
    public Scanner reset(@GuardSatisfied Scanner this);

    @Positive
    public Stream<String> tokens();

    @Positive
    class TokenSpliterator extends Spliterators.AbstractSpliterator<String> {

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super String> cons);
    @Positive
    }

    @Positive
    public Stream<MatchResult> findAll(Pattern pattern);

    @Positive
    public Stream<MatchResult> findAll(String patString);

    @Positive
    class FindSpliterator extends Spliterators.AbstractSpliterator<MatchResult> {

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super MatchResult> cons);
    @Positive
    }

    @Positive
    private static class PatternLRUCache {

    @Positive
        boolean hasName(Pattern p, String s);

    @Positive
        void moveToFront(Object[] oa, int i);

    @Positive
        Pattern forName(String name);
    @Positive
    }
    @Positive
}
