/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.ref.SoftReference;
import java.text.spi.BreakIteratorProvider;
import java.util.Locale;
import sun.util.locale.provider.LocaleProviderAdapter;
import sun.util.locale.provider.LocaleServiceProviderPool;

@AnnotatedFor({ "index" })
public abstract class BreakIterator implements Cloneable {

    protected BreakIterator() {
    }

    @Override
    public Object clone();

    @IntVal({ -1 })
    public static final int DONE;

    @GTENegativeOne
    public abstract int first();

    @GTENegativeOne
    public abstract int last();

    @GTENegativeOne
    public abstract int next(int n);

    @GTENegativeOne
    public abstract int next();

    @GTENegativeOne
    public abstract int previous();

    @GTENegativeOne
    public abstract int following(@NonNegative int offset);

    @GTENegativeOne
    public int preceding(@NonNegative int offset);

    public boolean isBoundary(@NonNegative int offset);

    @GTENegativeOne
    public abstract int current();

    public abstract CharacterIterator getText();

    public void setText(String newText);

    public abstract void setText(CharacterIterator newText);

    public static BreakIterator getWordInstance();

    public static BreakIterator getWordInstance(Locale locale);

    public static BreakIterator getLineInstance();

    public static BreakIterator getLineInstance(Locale locale);

    public static BreakIterator getCharacterInstance();

    public static BreakIterator getCharacterInstance(Locale locale);

    public static BreakIterator getSentenceInstance();

    public static BreakIterator getSentenceInstance(Locale locale);

    public static synchronized Locale[] getAvailableLocales();

    private static final class BreakIteratorCache {

        Locale getLocale();

        BreakIterator createBreakInstance();
    }
}
