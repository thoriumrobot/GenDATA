/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import com.sun.tools.javac.code.Kinds.Kind;
    @Positive
import com.sun.tools.javac.code.Scope.WriteableScope;
    @Positive
import com.sun.tools.javac.code.Symbol;
    @Positive
import com.sun.tools.javac.code.Symbol.ClassSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.Completer;
    @Positive
import com.sun.tools.javac.code.Symbol.CompletionFailure;
    @Positive
import com.sun.tools.javac.util.Context;

    @Positive
public class DeferredCompletionFailureHandler {

    @Positive
    protected static final Context.Key<DeferredCompletionFailureHandler> deferredCompletionFailureHandlerKey;

    @Positive
    public static DeferredCompletionFailureHandler instance(Context context);

    @Positive
    public final Handler userCodeHandler;

    @Positive
    public final Handler speculativeCodeHandler;

    @Positive
    public final Handler javacCodeHandler;

    @Positive
    protected DeferredCompletionFailureHandler(Context context) {
    @Positive
    }

    @Positive
    public Handler setHandler(Handler h);

    @Positive
    public void handleAPICompletionFailure(CompletionFailure cf);

    @Positive
    public void classSymbolCompleteFailed(ClassSymbol sym, Completer origCompleter);

    @Positive
    public void classSymbolRemoved(ClassSymbol sym);

    @Positive
    public boolean isDeferredCompleter(Completer c);

    @Positive
    public interface Handler {

    @Positive
        public void install();

    @Positive
        public void handleAPICompletionFailure(CompletionFailure cf);

    @Positive
        public void classSymbolCompleteFailed(ClassSymbol sym, Completer origCompleter);

    @Positive
        public void classSymbolRemoved(ClassSymbol sym);

    @Positive
        public void uninstall();
    @Positive
    }

    @Positive
    @UsesObjectEquals
    @Positive
    private class DeferredCompleter implements Completer {

    @Positive
        public DeferredCompleter(Completer origCompleter) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void complete(Symbol sym) throws CompletionFailure;
    @Positive
    }

    @Positive
    private static class FlipSymbolDescription {

    @Positive
        public final ClassSymbol sym;

    @Positive
        public Type type;

    @Positive
        public Kind kind;

    @Positive
        public WriteableScope members;

    @Positive
        public Completer completer;

    @Positive
        public FlipSymbolDescription(ClassSymbol sym, Completer completer) {
    @Positive
        }

    @Positive
        public void flip();
    @Positive
    }
    @Positive
}
