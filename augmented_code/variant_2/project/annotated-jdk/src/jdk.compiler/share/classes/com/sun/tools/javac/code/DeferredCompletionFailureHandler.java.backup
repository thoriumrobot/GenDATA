/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import java.util.HashMap;
import java.util.Map;
import com.sun.tools.javac.code.Kinds.Kind;
import com.sun.tools.javac.code.Scope.WriteableScope;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symbol.Completer;
import com.sun.tools.javac.code.Symbol.CompletionFailure;
import com.sun.tools.javac.util.Context;

public class DeferredCompletionFailureHandler {

    protected static final Context.Key<DeferredCompletionFailureHandler> deferredCompletionFailureHandlerKey;

    public static DeferredCompletionFailureHandler instance(Context context);

    public final Handler userCodeHandler;

    public final Handler speculativeCodeHandler;

    public final Handler javacCodeHandler;

    protected DeferredCompletionFailureHandler(Context context) {
    }

    public Handler setHandler(Handler h);

    public void handleAPICompletionFailure(CompletionFailure cf);

    public void classSymbolCompleteFailed(ClassSymbol sym, Completer origCompleter);

    public void classSymbolRemoved(ClassSymbol sym);

    public boolean isDeferredCompleter(Completer c);

    public interface Handler {

        public void install();

        public void handleAPICompletionFailure(CompletionFailure cf);

        public void classSymbolCompleteFailed(ClassSymbol sym, Completer origCompleter);

        public void classSymbolRemoved(ClassSymbol sym);

        public void uninstall();
    }

    @UsesObjectEquals
    private class DeferredCompleter implements Completer {

        public DeferredCompleter(Completer origCompleter) {
        }

        @Override
        public void complete(Symbol sym) throws CompletionFailure;
    }

    private static class FlipSymbolDescription {

        public final ClassSymbol sym;

        public Type type;

        public Kind kind;

        public WriteableScope members;

        public Completer completer;

        public FlipSymbolDescription(ClassSymbol sym, Completer completer) {
        }

        public void flip();
    }
}
