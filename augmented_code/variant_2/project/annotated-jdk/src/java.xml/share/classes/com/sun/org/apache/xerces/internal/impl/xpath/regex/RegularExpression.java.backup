/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xpath.regex;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.text.CharacterIterator;
import java.util.Locale;
import java.util.Stack;
import com.sun.org.apache.xerces.internal.util.IntStack;

public class RegularExpression implements java.io.Serializable {

    public boolean matches(char[] target);

    public boolean matches(char[] target, int start, int end);

    public boolean matches(char[] target, Match match);

    public boolean matches(char[] target, int start, int end, Match match);

    public boolean matches(String target);

    public boolean matches(String target, int start, int end);

    public boolean matches(String target, Match match);

    public boolean matches(String target, int start, int end, Match match);

    boolean matchAnchor(ExpressionTarget target, Op op, Context con, int offset, int opts);

    public boolean matches(CharacterIterator target);

    public boolean matches(CharacterIterator target, Match match);

    static abstract class ExpressionTarget {

        abstract char charAt(int index);

        abstract boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

        abstract boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    }

    static final class StringTarget extends ExpressionTarget {

        final void resetTarget(String target);

        final char charAt(int index);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    }

    static final class CharArrayTarget extends ExpressionTarget {

        final void resetTarget(char[] target);

        char charAt(int index);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    }

    static final class CharacterIteratorTarget extends ExpressionTarget {

        final void resetTarget(CharacterIterator target);

        final char charAt(int index);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    }

    static final class ClosureContext {

        @Pure
        boolean contains(int offset);

        void reset();

        void addOffset(int offset);
    }

    static final class Context {

        void reset(CharacterIterator target, int start, int limit, int nofclosures);

        void reset(String target, int start, int limit, int nofclosures);

        void reset(char[] target, int start, int limit, int nofclosures);

        synchronized void setInUse(boolean inUse);
    }

    void prepare();

    public RegularExpression(String regex) throws ParseException {
    }

    public RegularExpression(String regex, String options) throws ParseException {
    }

    public RegularExpression(String regex, String options, Locale locale) throws ParseException {
    }

    public void setPattern(String newPattern) throws ParseException;

    public void setPattern(String newPattern, Locale locale) throws ParseException;

    public void setPattern(String newPattern, String options) throws ParseException;

    public void setPattern(String newPattern, String options, Locale locale) throws ParseException;

    public String getPattern();

    public String toString();

    public String getOptions();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    boolean equals(String pattern, int options);

    public int hashCode();

    public int getNumberOfGroups();
}
