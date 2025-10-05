/*
    @Positive
 * Copyright (c) 2005, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

    @Positive
import org.checkerframework.checker.regex.qual.Regex;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ArrayList;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Date;
    @Positive
import java.util.List;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.regex.PatternSyntaxException;

    @Positive
@AnnotatedFor({ "regex" })
    @Positive
public abstract class RowFilter<M, I> {

    @Positive
    public enum ComparisonType {

    @Positive
        BEFORE, AFTER, EQUAL, NOT_EQUAL
    @Positive
    }

    @Positive
    protected RowFilter() {
    @Positive
    }

    @Positive
    public static <M, I> RowFilter<M, I> regexFilter(@Regex String regex, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> dateFilter(ComparisonType type, Date date, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> numberFilter(ComparisonType type, Number number, int... indices);

    @Positive
    public static <M, I> RowFilter<M, I> orFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    @Positive
    public static <M, I> RowFilter<M, I> andFilter(Iterable<? extends RowFilter<? super M, ? super I>> filters);

    @Positive
    public static <M, I> RowFilter<M, I> notFilter(RowFilter<M, I> filter);

    @Positive
    public abstract boolean include(Entry<? extends M, ? extends I> entry);

    @Positive
    public abstract static class Entry<M, I> {

    @Positive
        public Entry() {
    @Positive
        }

    @Positive
        public abstract M getModel();

    @Positive
        public abstract int getValueCount();

    @Positive
        public abstract Object getValue(int index);

    @Positive
        public String getStringValue(int index);

    @Positive
        public abstract I getIdentifier();
    @Positive
    }

    @Positive
    private abstract static class GeneralFilter<M, I> extends RowFilter<M, I> {

    @Positive
        @Override
    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);

    @Positive
        protected abstract boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class RegexFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class DateFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class NumberFilter<M, I> extends GeneralFilter<M, I> {

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        protected boolean include(Entry<? extends M, ? extends I> value, int index);
    @Positive
    }

    @Positive
    private static class OrFilter<M, I> extends RowFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }

    @Positive
    private static class AndFilter<M, I> extends OrFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }

    @Positive
    private static class NotFilter<M, I> extends RowFilter<M, I> {

    @Positive
        public boolean include(Entry<? extends M, ? extends I> value);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
