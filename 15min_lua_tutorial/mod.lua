local M = {}

local function sayMyName()
    print('Jimmy')
end

function M.sayHello()
    print('Why hello there')
    sayMyName()
end

return M
