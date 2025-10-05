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
import org.checkerframework.checker.i18nformatter.qual.I18nFormatFor;
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.text.DecimalFormat;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Date;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;

    @Positive
@AnnotatedFor({ "i18nformatter", "nullness" })
    @Positive
public class MessageFormat extends Format {

    @Positive
    public MessageFormat(String pattern) {
    @Positive
    }

    @Positive
    public MessageFormat(String pattern, Locale locale) {
    @Positive
    }

    @Positive
    public void setLocale(Locale locale);

    @Positive
    public Locale getLocale();

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void applyPattern(String pattern);

    @Positive
    public String toPattern();

    @Positive
    public void setFormatsByArgumentIndex(Format[] newFormats);

    @Positive
    public void setFormats(Format[] newFormats);

    @Positive
    public void setFormatByArgumentIndex(int argumentIndex, Format newFormat);

    @Positive
    public void setFormat(int formatElementIndex, Format newFormat);

    @Positive
    @Nullable
    @Positive
    public Format[] getFormatsByArgumentIndex();

    @Positive
    public Format[] getFormats();

    @Positive
    public final StringBuffer format(@Nullable Object @Nullable [] arguments, StringBuffer result, @Nullable FieldPosition pos);

    @Positive
    public static String format(@I18nFormatFor("#2") String pattern, @Nullable Object... arguments);

    @Positive
    public final StringBuffer format(Object arguments, StringBuffer result, FieldPosition pos);

    @Positive
    public AttributedCharacterIterator formatToCharacterIterator(Object arguments);

    @Positive
    public Object[] parse(@Nullable String source, ParsePosition pos);

    @Positive
    public Object[] parse(String source) throws ParseException;

    @Positive
    @Nullable
    @Positive
    public Object parseObject(String source, ParsePosition pos);

    @Positive
    public Object clone();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public static class Field extends Format.Field {

    @Positive
        protected Field(String name) {
    @Positive
        }

    @Positive
        @java.io.Serial
    @Positive
        protected Object readResolve() throws InvalidObjectException;

    @Positive
        public static final Field ARGUMENT;
    @Positive
    }
    @Positive
}
