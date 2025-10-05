/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.text;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.text.spi.BreakIteratorProvider;
    @Positive
import java.util.Locale;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.LocaleServiceProviderPool;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public abstract class BreakIterator implements Cloneable {

    @Positive
    protected BreakIterator() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    @IntVal({ -1 })
    @Positive
    public static final int DONE;

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int first();

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int last();

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int next(int n);

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int next();

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int previous();

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int following(@NonNegative int offset);

    @Positive
    @GTENegativeOne
    @Positive
    public int preceding(@NonNegative int offset);

    @Positive
    public boolean isBoundary(@NonNegative int offset);

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int current();

    @Positive
    public abstract CharacterIterator getText();

    @Positive
    public void setText(String newText);

    @Positive
    public abstract void setText(CharacterIterator newText);

    @Positive
    public static BreakIterator getWordInstance();

    @Positive
    public static BreakIterator getWordInstance(Locale locale);

    @Positive
    public static BreakIterator getLineInstance();

    @Positive
    public static BreakIterator getLineInstance(Locale locale);

    @Positive
    public static BreakIterator getCharacterInstance();

    @Positive
    public static BreakIterator getCharacterInstance(Locale locale);

    @Positive
    public static BreakIterator getSentenceInstance();

    @Positive
    public static BreakIterator getSentenceInstance(Locale locale);

    @Positive
    public static synchronized Locale[] getAvailableLocales();

    @Positive
    private static final class BreakIteratorCache {

    @Positive
        Locale getLocale();

    @Positive
        BreakIterator createBreakInstance();
    @Positive
    }
    @Positive
}
